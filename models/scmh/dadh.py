import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init

class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.hashing = nn.Sequential(
            nn.Linear(512, code_len, bias=True), 
            nn.Tanh()
            )

    def forward(self, x):
        x = self.hashing(x)
        return x


class TxtNet(nn.Module):
    def __init__(self, code_len):
        super(TxtNet, self).__init__()
        self.hashing = nn.Sequential(
            nn.Linear(512, code_len, bias=True),
            nn.Tanh()
            )


    def forward(self, x):
        x = self.hashing(x)
        return x
    
class DIS(torch.nn.Module):
    def __init__(self, code_len):
        super(DIS, self).__init__()
        self.hash_dis = nn.Sequential(
            nn.Linear(code_len, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 1, bias=True)
        )

        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, h):
        hash_score = self.hash_dis(h)
        return hash_score.squeeze()
    

def cos_distance(source, target):
    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)
    distances = torch.clamp(1 - cos_sim, 0)

    return distances


def get_triplet_mask(s_labels, t_labels, opt):
    flag = (opt.beta - 0.1) * opt.gamma
    batch_size = s_labels.shape[0]
    sim_origin = s_labels.mm(t_labels.t())
    sim = (sim_origin > 0).float()
    ideal_list = torch.sort(sim_origin, dim=1, descending=True)[0]
    ph = torch.arange(0., batch_size) + 2
    ph = ph.repeat(1, batch_size).reshape(batch_size, batch_size)
    th = torch.log2(ph).cuda()
    Z = (((2 ** ideal_list - 1) / th).sum(axis=1)).reshape(-1, 1)
    sim_origin = 2 ** sim_origin - 1
    sim_origin = sim_origin / Z

    i_equal_j = sim.unsqueeze(2)
    i_equal_k = sim.unsqueeze(1)
    sim_pos = sim_origin.unsqueeze(2)
    sim_neg = sim_origin.unsqueeze(1)
    weight = (sim_pos - sim_neg) * (flag + 0.1)
    mask = i_equal_j * (1 - i_equal_k) * (flag + 0.1)

    return mask, weight


class TripletLoss(nn.Module):
    def __init__(self, opt, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.reduction = reduction
        self.opt = opt

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        pairwise_dist = cos_distance(source, target)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask, weight = get_triplet_mask(s_labels, t_labels, self.opt)
        if self.opt.alpha == 10:
            triplet_loss = 10 * weight * mask * triplet_loss
        else:
            triplet_loss = mask * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = triplet_loss.clamp(0)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss.gt(1e-16).float()
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss