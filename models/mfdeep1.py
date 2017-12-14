import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.variational_biased_embedding import VariationalBiasedEmbedding


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class MFDeep1(nn.Module):
    is_train = True

    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1., loss=nn.MSELoss):
        super(MFDeep1, self).__init__()
        self.embed_user = VariationalBiasedEmbedding(n_users, n_dim, lb=lub,
                                                     lv=luv, n_obs=n_obs)
        self.embed_item = VariationalBiasedEmbedding(n_items, n_dim, lb=lib,
                                                     lv=liv, n_obs=n_obs)
        self.lin1 = nn.Linear(n_dim, n_dim)
        self.lin2 = nn.Linear(n_dim * 3, 2)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        self.n_obs = n_obs
        self.lossf = loss()
        self.ad1 = nn.AlphaDropout(p=0.1)
        self.ad2 = nn.AlphaDropout(p=0.1)
        self.lin1.weight.data *= 1e-9
        self.glob_bias.data[:] = 0.3

    def forward(self, u, i):
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        # x0 = torch.cat(vu * vi, dim=1)
        x0 = vu * vi
        x1 = self.ad1(selu(self.lin1(x0)))
        # x2 = self.ad2(selu(self.lin2(x1)))
        # x2 = x2.sum(dim=1).squeeze()
        logodds = bias + bi + bu + x0.sum(dim=1) + x1.sum(dim=1)
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = self.lossf(prediction, target)
        reg = (self.embed_user.prior() + self.embed_item.prior()) / n_batches
        return llh + reg
