import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Exactly the same as the MF model, but with 2nd
# order polynomial effect on age


class BiasedEmbedding(nn.Module):
    def __init__(self, n_feat, n_dim, lv=1.0, lb=1.0):
        super(BiasedEmbedding, self).__init__()
        self.vect = nn.Embedding(n_feat, n_dim)
        self.bias = nn.Embedding(n_feat, 1)
        self.lv = lv
        self.lb = lb

    def __call__(self, index):
        return self.bias(index).squeeze(), self.vect(index)

    def prior(self):
        loss = (self.vect.weight.sum()**2.0 * self.lv +
                self.bias.weight.sum()**2.0 * self.lb)
        return loss


class MFPoly2(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1.):
        super(MFPoly2, self).__init__()
        self.embed_user = BiasedEmbedding(n_users, n_dim, lb=lub, lv=luv)
        self.embed_item = BiasedEmbedding(n_items, n_dim, lb=lib, lv=liv)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        # 2nd order polynomial = pass through 2 linear layers it linearly
        # maps from a scalar (dim=1) of age to a scalar log odds (dim=1)
        self.age1 = nn.Linear(1, 1)
        self.age2 = nn.Linear(1, 1)
        self.n_obs = n_obs

    def forward(self, u, i, a):
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        intx = (vu * vi).sum(dim=1)
        age_effect = self.age2(self.age1(a.view(len(u), 1))).squeeze()
        logodds = bias + bi + bu + intx + age_effect
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = F.binary_cross_entropy_with_logits(prediction, target)
        reg = (self.embed_user.prior() + self.embed_item.prior()) / n_batches
        return llh + reg
