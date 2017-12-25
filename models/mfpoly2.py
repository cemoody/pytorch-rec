import torch
from torch import nn
from torch.nn.parameter import Parameter

from biased_embedding import BiasedEmbedding

# Exactly the same as the MF model, but with 2nd
# order polynomial effect on age


class MFPoly2(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1., loss=nn.MSELoss):
        super(MFPoly2, self).__init__()
        self.embed_user = BiasedEmbedding(n_users, n_dim, lb=lub, lv=luv)
        self.embed_item = BiasedEmbedding(n_items, n_dim, lb=lib, lv=liv)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        # maps from a scalar (dim=2) of age and age^2
        # to a scalar log odds (dim=1)
        self.age = nn.Linear(2, 1)
        self.n_obs = n_obs
        self.lossf = loss()

    def forward(self, u, i, a):
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        intx = (vu * vi).sum(dim=1)
        av = a.view(len(u), 1)
        age_effect = self.age(torch.cat([av, av**2], dim=1)).squeeze()
        logodds = bias + bi + bu + intx + age_effect
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = self.lossf(prediction, target)
        reg = (self.embed_user.prior() + self.embed_item.prior()) / n_batches
        return llh + reg
