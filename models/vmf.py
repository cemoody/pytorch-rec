import torch
from torch import nn
from torch.nn.parameter import Parameter

from models.variational_biased_embedding \
        import VariationalBiasedEmbedding as VBE


class VMF(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1., loss=nn.MSELoss):
        super(VMF, self).__init__()
        self.embed_user = VBE(n_users, n_dim, lb=lub, lv=luv)
        self.embed_item = VBE(n_items, n_dim, lb=lib, lv=liv)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        self.n_obs = n_obs
        self.lossf = loss()

    def forward(self, input):
        u, i = input
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        intx = (vu * vi).sum(dim=1)
        logodds = bias + bi + bu + intx
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = self.lossf(prediction, target)
        reg = (self.embed_user.prior() + self.embed_item.prior()) / n_batches
        return llh + reg
