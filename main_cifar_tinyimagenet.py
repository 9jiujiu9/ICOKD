import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
import csv

from dataset.tinyimagenet import get_tinyimagenet_dataloaders

import models
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
from itertools import chain

from bisect import bisect_right
import time
import math

import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

from dataset.class_sampler import MPerClassSampler

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='tinyimagenet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet32_n', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight deacy')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 180, 210], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet32.pth.tar', type=str, help='resume checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--evaluate-checkpoint', default='./checkpoint/resnet32_best.pth.tar', type=str,
                    help='evaluate checkpoint')
parser.add_argument('--number-net', type=int, default=2, help='number of networks')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')

parser.add_argument('--kd_T', type=float, default=3, help='temperature of KL-divergence')
parser.add_argument('--tau', default=0.1, type=float, help='temperature for contrastive distribution')
parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for VCL')
parser.add_argument('--gamma', type=float, default=1.0, help='weight balance for Soft VCL')
# parser.add_argument('--beta', type=float, default=0., help='weight balance for ICL')
parser.add_argument('--lam', type=float, default=1., help='weight balance for Soft ICL')
parser.add_argument('--feat-dim', default=512, type=int, help='feature dimension')
parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax', 'queue'])
parser.add_argument('--nce_k', default=500, type=int, help='number of negative samples for NCE') # 

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.isdir('./result/'):
    os.makedirs('./result/')

log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + \
          'seed' + str(args.manual_seed) + '.txt'

with open(log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

if args.dataset == 'cifar100':
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                    [0.2675, 0.2565, 0.2761])
                                            ]))

    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                    [0.2675, 0.2565, 0.2761]),
                                            ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          sampler=MPerClassSampler(labels=trainset.targets, m=2,
                                                                   batch_size=args.batch_size,
                                                                   length_before_new_iter=len(trainset.targets)),
                                          pin_memory=(torch.cuda.is_available()))

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))

elif args.dataset == 'cifar10':
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                    [0.2470, 0.2435, 0.2616])
                                            ]))

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                    [0.2470, 0.2435, 0.2616]),
                                            ]))
                                            
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          sampler=MPerClassSampler(labels=trainset.targets, m=16,
                                                                   batch_size=args.batch_size,
                                                                   length_before_new_iter=len(trainset.targets)),
                                          pin_memory=(torch.cuda.is_available()))

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))

elif args.dataset == 'tinyimagenet':
    trainloader, testloader, n_data = get_tinyimagenet_dataloaders(root = args.data, batch_size=args.batch_size,
                                                                        val_batch_size=args.batch_size,
                                                                           num_workers=args.num_workers
                                                                        )
    num_classes = 200
# --------------------------------------------------------------------------------------------
# if args.dataset == 'cifar100':
#         trainloader, testloader, n_data = get_cifar100_dataloaders_sample(batch_size=args.batch_size,
#                                                                            num_workers=args.num_workers,
#                                                                            k=args.nce_k,
#                                                                            mode=args.mode)
       
#         n_cls = 100
# else:
#     raise NotImplementedError(args.dataset)

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes, number_net=args.number_net)
net.eval()
resolution = (1, 3, 64, 64)
print('Arch: %s, Params: %.2fM, FLOPs: %.2fG'
      % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, resolution) / 1e9))
del (net)

net = model(num_classes=num_classes, number_net=args.number_net).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True


def train(epoch, criterion_list, optimizer):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_logits = AverageMeter('train_loss_logits', ':.4e')
    # train_loss_f = AverageMeter('train_loss_logit_f', ':.4e')
    # train_loss_vcl = AverageMeter('train_loss_vcl', ':.4e')
    # train_loss_icl = AverageMeter('train_loss_icl', ':.4e')
    # train_loss_soft_vcl = AverageMeter('train_loss_soft_vcl', ':.4e')
    # train_loss_soft_icl = AverageMeter('train_loss_soft_icl', ':.4e')

    top1_num = [0] * args.number_net
    top5_num = [0] * args.number_net
    total = [0] * args.number_net

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]
    # criterion_kd = criterion_list[3]
    # criterion_afd2 = criterion_list[4]
    # criterion_afd3 = criterion_list[5]
    LSLoss = LabelSmoothingLoss(classes=100)

    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_start_time = time.time()
        inputs = inputs.float().cuda()
        targets = targets.cuda()

        # r = np.random.rand(1)
        # if args.beta > 0 and r < args.cutmix_prob:
        #     # generate mixed sample
        #     lam = np.random.beta(args.beta, args.beta)
        #     rand_index = torch.randperm(input.size()[0]).cuda()
        #     target_a = targets
        #     target_b = targets[rand_index]
        #     bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        #     input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        #     # adjust lambda to exactly match pixel ratio
        #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

        # r = np.random.rand(1)
        # if r < args.cutmix_prob:
        #     # generate mixed sample
        #     lam = np.random.beta(args.beta, args.beta)
        #     rand_index = torch.randperm(inputs.size()[0]).cuda()
        #     target_a = targets
        #     target_b = targets[rand_index]
        #     bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        #     # batch_idx_updated= inputs.clone()
        #     inputs[:, bbx1:bbx2, bby1:bby2, :] = inputs[rand_index, bbx1:bbx2, bby1:bby2, :]
        #     # adjust lambda to exactly match pixel ratio
        #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        #     targets = target_a * lam + target_b * (1. - lam)

        optimizer.zero_grad()
        logits, embedding, f = net(inputs)

        # a=torch.tensor([f[0]])
        # # print(a.shape)
        # print(a.size())

        # b=torch.tensor(f)
        # # print(b.shape)
        # print(b.size())

        # c=torch.tensor([x_totalfea[0]])
        # # print(c.shape)
        # print(c.size())

        loss_cls = torch.tensor(0.).cuda()
        loss_logit_kd = torch.tensor(0.).cuda()
        # MSEloss = nn.MSELoss(reduction='mean').cuda()
        # loss_f = torch.tensor(0.).cuda()

        for i in range(len(logits)):
            loss_cls = loss_cls + criterion_ce(logits[i], targets)

 
        # ensemble_logits = 0.
        # for i in range(len(logits)):
        #     ensemble_logits = ensemble_logits + logits[i]
        # ensemble_logits = ensemble_logits / (len(logits))

        # ensemble_logits = ensemble_logits.detach()

        # # ensemble_logits = fea_logit.detach()
        # # loss_f = loss_f + criterion_fea(f[0], f[3])
        # # loss_f = loss_f + criterion_fea(f[0], f[3])
        # # loss_f = loss_f + criterion_fea(f[1], f[4])
        # loss_f = loss_f + criterion_fea(f[2], f[5])
        
        
        # loss_fea = loss_fea + criterion_kd(f[2], f[5], index, contrast_idx)

        # # fea_logit=0
        # # for i in range(len(f_logits)):
        # #     fea_logit=fea_logit+f_logits[i]
        # # fea_logit=fea_logit/len(f_logits)
        # #
        # # fea_logit=fea_logit.detach()
        # # if epoch % args.number_net == 0:
        # #     f_to = x_t[0]
        # # elif epoch % args.number_net == 1:
        # #     f_to = x_t[1]
        # # else:
        # #     f_to = x_t[2]
        
        loss_logit_kd = loss_logit_kd + criterion_div(logits[0], logits[1])
        loss_logit_kd = loss_logit_kd + criterion_div(logits[1], logits[0])
                
        # else:
        #     loss_logit_kd = loss_logit_kd + 1/3 * criterion_div(logits[0], ensemble_logits)
        #     loss_logit_kd = loss_logit_kd + 1/3 * criterion_div(logits[1], ensemble_logits)
        #     loss_logit_kd = loss_logit_kd + 1/3 * criterion_div(logits[2], ensemble_logits)
        #     loss_logit_kd = loss_logit_kd + 0.5 * criterion_kd(f[0], f[1], index, contrast_idx)
        #     loss_logit_kd = loss_logit_kd + 0.5 * criterion_kd(f[1], f[2], index, contrast_idx)
            # else:
            #     loss_logit_kd = loss_logit_kd + criterion_div(logits[-1], ensemble_logits)
            # loss_logit_kd = loss_logit_kd + criterion_div(fea_logit, ensemble_logits)
            # loss_logit_kd = loss_logit_kd + criterion_div(ensemble_logits, f_logits)
        # loss_logit_kd = loss_logit_kd + criterion_div(ensemble_logits, x_t[0])

        # for i in range(len(f_logits)-1):
        #     loss_logit_kd = loss_logit_kd + criterion_div(f_logits[i], f_logits[i+1])
        #     loss_logit_kd = loss_logit_kd + criterion_div(f_logits[i+1], f_logits[i])
        # #     loss_logit_kd = loss_logit_kd + criterion_div(logits[i], f_logits[i])
        # #     loss_logit_kd = loss_logit_kd + criterion_div(logits[i], f_logits[i])
        # # loss_logit_kd = loss_logit_kd + criterion_div(logits[i+1], f_logits[i+1])
        # # loss_logit_kd = loss_logit_kd + criterion_div( f_logits[i + 1],logits[i+1])
        #     loss_logit_kd = loss_logit_kd + criterion_div(logits[i], f_logits[i])
        # loss_logit_kd = loss_logit_kd + criterion_div(logits[i + 1], f_logits[i + 1])

        # if args.logit_distill:
        #     for i in range(args.number_net):
        #         # logits[i]=F.interpolate(logits[i],size=[100])
        #         # im1_torch = torch.from_numpy(logits[i].unsqueeze(0))
        #         # torch_resize = Resize([100]) # 定义Resize类对象
        #         # x = torch_resize(logits[i])
        #         feature_logits=torch.reshape(feature_logits,[64, 100])
        #         logits[i]=logits[i].cuda()
        #         feature_logits=feature_logits.cuda()
        #         loss_logit_kd = loss_logit_kd + criterion_div(logits[i], feature_logits)
        #     loss_logit_kd=loss_logit_kd+criterion_div(feature_logits, ensemble_logits)

        # for i in range(args.number_net):
        #     for j in range(i*3,i*3+3):
        #         loss_fd = loss_fd + criterion_fd(f[j], x_totalfea[i])

        # loss_fea = criterion_fea(embedding, targets,f_t)
        # fea_logit kl
        # loss_vcl, loss_soft_vcl, loss_icl, loss_soft_icl = criterion_fea(embedding, targets,f_t)
        # loss_mcl = args.alpha * loss_vcl + args.gamma * loss_soft_vcl \
        #        + args.beta * loss_icl + args.lam * loss_soft_icl
        
        # loss = loss_cls + loss_logit_kd + loss_mcl
        # f_t=f_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # for i in range(len(f)):
        #     g=torch.Tensor(f[i])
        # g=g.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # for i in range(args.number_net):
        #     for j in range(i*3,i*3+3):
        #         loss_fea += criterion_feature(g[j], f_t[i].detach())

        # for i in range(args.number_net):
        #         loss_fea += criterion_feature(f_t[i], x_totalfea.detach())

# 特征蒸馏
#         for i in range(3):
#             f[i] = f[i].unsqueeze(-1).unsqueeze(-1)
#             f[i + 3] = f[i + 3].unsqueeze(-1).unsqueeze(-1)
#
#             if i%3==0:
#                 loss_fea = loss_fea + args.gamma * (criterion_list[3](f[i], f[i + 3].detach()))
#                 loss_fea = loss_fea + args.gamma * (criterion_list[3](f[i + 3], f[i].detach()))
#             elif i%3==1:
#                 loss_fea = loss_fea + args.gamma * (criterion_list[4](f[i], f[i + 3].detach()))
#                 loss_fea = loss_fea + args.gamma * (criterion_list[4](f[i + 3], f[i].detach()))
#             else:
#                 loss_fea = loss_fea + args.gamma * (criterion_list[5](f[i], f[i + 3].detach()))
#                 loss_fea = loss_fea + args.gamma * (criterion_list[5](f[i + 3], f[i].detach()))
        # x_totalfea=x_totalfea.unsqueeze(-1).unsqueeze(-1)
        # for i in range(args.number_net):
        #     f_t[i]=f_t[i].unsqueeze(-1).unsqueeze(-1)
        #     loss_fea += args.gamma*(criterion_list[3](f_t[i], x_totalfea.detach()))

        # for m in range(args.number_net):
        #     for j in range(len(f)):
        #         f[j]=f[j].unsqueeze(-1).unsqueeze(-1)
        #         total=f_t[m].unsqueeze(-1).unsqueeze(-1)
        #         # con=nn.Conv2d(1200, 128, kernel_size=1)
        #         # x_totalf=con(x_totalfea)
        #         loss_fea += args.gamma*(criterion_list[3](f[j], total.detach()))

        # f_t[0]=f_t[0].unsqueeze(-1).unsqueeze(-1)
        # for i in range(args.number_net-1):
        #     # f_t[i] = f_t[i].unsqueeze(-1).unsqueeze(-1)
        #     f_t[i+1]=f_t[i+1].unsqueeze(-1).unsqueeze(-1)
        #     # con=nn.Conv2d(1200, 128, kernel_size=1)
        #     # x_totalf=con(x_totalfea)
        #     loss_fea = loss_fea + args.gamma * (criterion_list[3](f_t[i], f_t[i+1].detach()))
        #     loss_fea = loss_fea + args.gamma * (criterion_list[3](f_t[i+1], f_t[i].detach()))
        # for i in range(args.number_net):
        #     embedding[i]=embedding[i].unsqueeze(-1).unsqueeze(-1)
        #     x_totalfea=x_totalfea.unsqueeze(-1).unsqueeze(-1)
        #     loss_fea += criterion_feature(embedding[i], x_totalfea.detach())

        loss = args.gamma * loss_cls + args.alpha * loss_logit_kd / (args.number_net - 1)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_logits.update(loss_logit_kd.item(), inputs.size(0))
        # train_loss_angle.update(angle_loss.item(), inputs.size(0))
        # train_loss_f.update(loss_f.item(), inputs.size(0))
        # train_loss_vcl.update(args.alpha * loss_vcl.item(), inputs.size(0))
        # train_loss_soft_vcl.update(args.gamma * loss_soft_vcl.item(), inputs.size(0))
        # train_loss_icl.update(args.beta * loss_icl.item(), inputs.size(0))
        # train_loss_soft_icl.update(args.lam * loss_soft_icl.item(), inputs.size(0))

        for i in range(len(logits)):
            top1, top5 = correct_num(logits[i], targets, topk=(1, 5))
            top1_num[i] += top1
            top5_num[i] += top5
            total[i] += targets.size(0)

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time() - batch_start_time, (top1_num[0] / total[0]).item()))

    acc1 = [round((top1_num[i] / total[i]).item(), 4) for i in range(args.number_net)]
    acc5 = [round((top5_num[i] / total[i]).item(), 4) for i in range(args.number_net)]

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                '\n Train_loss:{:.5f}'
                '\t Train_loss_cls:{:.5f}'
                '\t Train_loss_logits:{:.5f}'
                # '\t Train_loss_f:{:.5f}'
                # '\t Train_loss_soft_vcl:{:.5f}'
                # '\t Train_loss_icl:{:.5f}'
                # '\t Train_loss_soft_icl:{:.5f}'
                '\n Train top-1 accuracy: {} \n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss.avg,
                        train_loss_cls.avg,
                        train_loss_logits.avg,
                        # train_loss_f.avg,
                        # train_loss_vcl.avg,
                        # train_loss_soft_vcl.avg,
                        # train_loss_icl.avg,
                        # train_loss_soft_icl.avg,
                        str(acc1)))


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def test(epoch, criterion_ce):
    net.eval()
    global best_acc
    test_loss_cls = AverageMeter('test_loss_cls', ':.4e')

    top1_num = [0] * (args.number_net + 1)
    top5_num = [0] * (args.number_net + 1)
    total = [0] * (args.number_net + 1)

    LSLoss = LabelSmoothingLoss(classes=100)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_start_time = time.time()
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, embedding, f = net(inputs)

            loss_cls = 0.
            ensemble_logits = 0.
            for i in range(len(logits)):
                loss_cls = loss_cls + criterion_ce(logits[i], targets)
            for i in range(len(logits)):
                ensemble_logits = ensemble_logits + logits[i]

            test_loss_cls.update(loss_cls, inputs.size(0))

            for i in range(args.number_net + 1):
                if i == args.number_net:
                    top1, top5 = correct_num(ensemble_logits, targets, topk=(1, 5))
                else:
                    top1, top5 = correct_num(logits[i], targets, topk=(1, 5))
                top1_num[i] += top1
                top5_num[i] += top5
                total[i] += targets.size(0)

            print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}'.format(epoch, batch_idx, len(testloader),
                                                                      time.time() - batch_start_time))

        acc1 = [round((top1_num[i] / total[i]).item(), 4) for i in range(args.number_net + 1)]
        acc5 = [round((top5_num[i] / total[i]).item(), 4) for i in range(args.number_net + 1)]

        with open(log_txt, 'a+') as f:
            f.write('Test epoch:{}\t Test_loss_cls:{:.5f}\t Test top-1 accuracy:{}\n'
                    .format(epoch, test_loss_cls.avg, str(acc1)))

        print('Test epoch:{}\t Test top-1 accuracy:{}\n'.format(epoch, str(acc1)))

    return max(acc1[:-1])

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_ce = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)

    if args.evaluate:
        print('Evaluate pre-trained weights from: {}'.format(args.evaluate_checkpoint))
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_ce)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)

        data = torch.randn(1, 3, 64, 64).cuda()
        net.eval()
        logits, embedding,f= net(data)

        # args.rep_dim = []
        # args.rep_dimen = []
        # args.rep_dime = []
        # # args.rep_dimtotal=[]
        # for x in embedding:
        #     args.rep_dim.append(x.size(1))
        # for y in f_t:
        #     args.rep_dimen.append(y.size(1))
        # for m in c:
        #     args.rep_dime.append(m.size(1))
        # args.rep_dimtotal.append(x_totalfea.size(1))

        # criterion_fea = Sup_MCL_Loss(args)
        # trainable_list.append(criterion_fea.embed_list)
        # criterion_fea = Floss()

        # args.n_data = n_data
        # criterion_kd = CRCDLoss(args)
        # trainable_list.append(criterion_kd.embed_s)
        # trainable_list.append(criterion_kd.embed_t)
        # s_channels  = embedding[i].size(1)
        # t_channels  = x_totalfea.size(1)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_ce)
        criterion_list.append(criterion_div)
        # criterion_list.append(AFD(f[0].size(1), args.att_f))
        # criterion_list.append(AFD(f[1].size(1), args.att_f))
        # criterion_list.append(AFD(f[2].size(1), args.att_f))
        criterion_list.cuda()

        optimizer = optim.SGD(trainable_list.parameters(),
                                      lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        if args.resume:
            print('Resume pre-trained weights from: {}'.format(args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
        
        # acc_csv_title = ['Epoch_idx', 'best_acc']

        # with open('./result/'+'/acc.csv', 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(acc_csv_title)

        for epoch in range(start_epoch, args.epochs):
            # acc_csv_result = []
            # acc_csv_result.append('Epoch'+str(epoch))
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_ce)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            # acc_csv_result.append(best_acc*100)

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

            # with open('./result/'+'/acc.csv', 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     # for idx, each in enumerate(acc_csv_result):
            #     # print(idx, each)
            #     # writer.writerow([each])
            #     writer.writerow(acc_csv_result)

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_ce)

        with open(log_txt, 'a+') as f:
            f.write('Test top-1 best_accuracy: {} \n'.format(top1_acc))
        print('Test top-1 best_accuracy: {} \n'.format(top1_acc))


