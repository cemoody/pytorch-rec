import torch
from torch import nn
from torch.nn import Parameter

from models.biased_embedding import BiasedEmbedding

# Exactly the same as the MF model, but with 2nd
# order polynomial effect on age


class MFPoly2(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1., loss=nn.MSELoss):
        super(MFPoly2, self).__init__()
        self.embed_user = BiasedEmbedding(n_users, n_dim, lb=lub, lv=luv)
        self.embed_item = BiasedEmbedding(n_items, n_dim, lb=lib, lv=liv)
        self.glob_bias = Parameter(torch.FloatTensor([0.01]))
        # self.glob_bias.grad = torch.FloatTensor([0.0])
        # maps from a scalar (dim=2) of age and age^2
        # to a scalar log odds (dim=1)
        self.frame = nn.Linear(2, 1)
        self.n_obs = n_obs
        self.lossf = loss()

    def forward(self, input):
        u, i, f = input
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        intx = (vu * vi).sum(dim=2).sum(dim=1)
        fv = f.view(len(u), 1)
        frame_effect = self.frame(torch.cat([fv, fv**2], dim=1)).squeeze()
        logodds = bias + bi + bu + intx + frame_effect
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = self.lossf(prediction, target)
        reg = (self.embed_user.prior() + self.embed_item.prior()) / n_batches
        return llh + reg
