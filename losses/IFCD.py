import torch
from torch import nn
from .IFC import IC
import torch.nn.functional as F

eps = 1e-7


class IFCDLoss(nn.Module):
    """CRCD Loss function
    
    Args:
        opt.embed_type: fc;nofc;nonlinear
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, args):
        super(IFCDLoss, self).__init__()
        # self.emd_fc_type = opt.embed_type
        # print("fc_type: {} ".format(self.emd_fc_type))
        # if self.emd_fc_type == "nofc":
        #     assert opt.s_dim == opt.t_dim
        #     opt.feat_dim = opt.s_dim
        self.dim = args.feat_dim
        self.feat_dim = args.feat_dim
        self.s_dim = args.feat_dim
        self.t_dim = args.feat_dim
        self.n_data =  args.n_data
        self.nce_k = args.nce_k
        self.embed_s = Embed(self.dim, self.s_dim, self.feat_dim)
        self.embed_t = Embed(self.dim, self.t_dim, self.feat_dim)
        self.contrast = IC(self.feat_dim, self.n_data, self.nce_k, 0.05, 0.5)
        self.kl = KLDiv(T=args.kd_T)
        # self.criterion_t = ContrastLoss(opt.n_data)
        # self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        There may be some learnable parameters in embedding layer in teacher side, 
        similar to crd, we also calculate the crcd loss over both the teacher and the student side.
        However, if the emd_fc_type=="nofc", then the 't_loss' term can be omitted.
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s1, out_t1 = self.contrast(f_s, f_t, idx, contrast_idx)
        out_s2, out_t2 = self.contrast(f_t, f_s, idx, contrast_idx)
        # out_s1 = self.contrast(f_s, f_t, idx, contrast_idx)
        # out_s2 = self.contrast(f_t, f_s, idx, contrast_idx)
        # s_loss = self.criterion_s(out_s)
        # t_loss = self.criterion_t(out_t)
        # loss = s_loss + t_loss
        
        loss = 0
        # loss += F.smooth_l1_loss(out_s, out_t, reduction='mean')
        # loss += F.smooth_l1_loss(out_t, out_s, reduction='mean')

        loss += self.kl(out_s1, out_t1)
        loss += self.kl(out_s2, out_t2)
        # loss += self.kl(out_s1, out_s2)
        # loss += self.kl(out_s2, out_s1)
        return loss


class ContrastLoss(nn.Module):
    ''' the contrastive loss is not critical  ''' 
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):    
        bsz = x.shape[0]
        m = x.size(1) - 1

        # 'loss old'
        # noise distribution
        Pn = 1 / float(self.n_data)
        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        # 'loss new'
        # P_pos = x.select(1, 0) # 64, 1
        # log_P= torch.log(P_pos)
        # # P_neg = x.narrow(1, 1, m) # bs, K, 1
        # # log_N= torch.log(P_pos + P_neg.sum(1))
        # log_N= torch.log(x.sum(1))
        # loss = ((- log_P.sum(0) + log_N.sum(0)) / bsz)

        
        return loss[0]

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':   
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha   
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)  
            
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)  

        return x * gate

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim, dim_in=1024, dim_out=128,emd_fc_type='nonlinear'):
        super(Embed, self).__init__()
        self.gct = GCT(dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if emd_fc_type == "linear":
            self.linear = nn.Linear(dim_in, dim_out)
        elif emd_fc_type == "nonlinear":
            self.linear = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, dim_out)
            )
        elif emd_fc_type == "nofc":
            self.linear = nn.Sequential()
        else:
            raise NotImplementedError(emd_fc_type)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.gct(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KLDiv, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss