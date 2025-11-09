import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np

import models
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
from itertools import chain

from bisect import bisect_right
import time
import csv

import numpy as np
import torch.nn as nn

from dataset.tinyimagenet import get_tinyimagenet_dataloaders_sample
from losses.IFCD import IFCDLoss

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

if args.dataset == 'tinyimagenet':
        trainloader, testloader, n_data = get_tinyimagenet_dataloaders_sample(batch_size=args.batch_size,
                                                                           num_workers=args.num_workers,
                                                                           k=args.nce_k,
                                                                           mode=args.mode)
        num_classes = 200
        n_cls = 200
else:
    raise NotImplementedError(args.dataset)

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
    train_loss_logit_kd = AverageMeter('train_loss_logit_kd', ':.4e')
    train_loss_fea = AverageMeter('train_loss_logit_fea', ':.4e')

    top1_num = [0] * args.number_net
    top5_num = [0] * args.number_net
    total = [0] * args.number_net

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    net.train()
    for batch_idx, (inputs, targets, index, contrast_idx) in enumerate(trainloader):
        batch_start_time = time.time()
        inputs = inputs.float().cuda()
        targets = targets.cuda()
        index = index.cuda()
        contrast_idx = contrast_idx.cuda()

        optimizer.zero_grad()
        logits, embedding, f = net(inputs)


        loss_cls = torch.tensor(0.).cuda()
        loss_logit_kd = torch.tensor(0.).cuda()
        loss_fea = torch.tensor(0.).cuda()
        

        for i in range(len(logits)):
            loss_cls = loss_cls + criterion_ce(logits[i], targets)

        loss_fea = loss_fea + criterion_kd(f[2], f[5], index, contrast_idx)
        
        criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')
        loss_t_list = [criterion_cls_lc(logit_s, targets) for logit_s in logits]
        loss_t = torch.stack(loss_t_list, dim=0)
        attention = (1.0 - F.softmax(loss_t, dim=0)) / (args.number_net - 1)
        loss_div_list = []
        for i in range(len(logits)-1):
            loss_div_list.append(0.5 * (criterion_div(logits[i], logits[i+1],is_ca=True) + criterion_div(logits[i+1], logits[i],is_ca=True)))
        loss_div_list.append(0.5 * (criterion_div(logits[i+1], logits[0],is_ca=True) + criterion_div(logits[i+1], logits[0],is_ca=True)))

        loss_div = torch.stack(loss_div_list, dim=0)
        bsz = loss_div.shape[1]
        loss_logit_kd = (torch.mul(attention, loss_div).sum()) / (1.0*bsz*args.number_net) 

        loss = args.gamma * loss_cls + args.alpha * loss_logit_kd + args.beta * loss_fea

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_logit_kd.update(loss_logit_kd.item(), inputs.size(0))
        
        train_loss_fea.update(loss_fea.item(), inputs.size(0))

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
                '\t Train_loss_logit_kd:{:.5f}'
                '\n Train top-1 accuracy: {} \n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss.avg,
                        train_loss_cls.avg,
                        train_loss_logit_kd.avg,
                        train_loss_fea.avg,
                        str(acc1)))


def test(epoch, criterion_ce):
    net.eval()
    global best_acc
    test_loss_cls = AverageMeter('test_loss_cls', ':.4e')

    top1_num = [0] * (args.number_net + 1)
    top5_num = [0] * (args.number_net + 1)
    total = [0] * (args.number_net + 1)

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
        logits, embedding, f = net(data)

        args.n_data = n_data
        criterion_kd = IFCDLoss(args)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_ce)
        criterion_list.append(criterion_div)
        criterion_list.append(criterion_kd)
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
        
        acc_csv_title = ['Epoch_idx', 'best_acc']

        with open('./result/'+'/acc.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(acc_csv_title)

        for epoch in range(start_epoch, args.epochs):
            acc_csv_result = []
            acc_csv_result.append('Epoch'+str(epoch))
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
            
            acc_csv_result.append(best_acc*100)

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

            with open('./result/'+'/acc.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                # for idx, each in enumerate(acc_csv_result):
                # print(idx, each)
                # writer.writerow([each])
                writer.writerow(acc_csv_result)

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


