import torch
from torch import nn
from torch.nn import Parameter

from models.biased_embedding import BiasedEmbedding


class MF(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1., loss=nn.MSELoss):
        super(MF, self).__init__()
        self.embed_user = BiasedEmbedding(n_users, n_dim, lb=lub, lv=luv)
        self.embed_item = BiasedEmbedding(n_items, n_dim, lb=lib, lv=liv)
        self.glob_bias = Parameter(torch.FloatTensor([0.01]))
        self.n_obs = n_obs
        self.lossf = loss()

    def forward(self, u, i):
        u, i = u.squeeze(), i.squeeze()
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        intx = (vu * vi).sum(dim=1)
        logodds = (bias + bi + bu + intx).squeeze()
        return logodds

    def loss(self, prediction, target):
        # average likelihood loss per example
        ex_llh = self.lossf(prediction, target)
        if self.training:
            # regularization penalty summed over whole model
            epoch_reg = (self.embed_user.prior() + self.embed_item.prior())
            # penalty should be computer for a single example
            ex_reg = epoch_reg * 1.0 / self.n_obs
        else:
            ex_reg = 0.0
        return ex_llh + ex_reg
