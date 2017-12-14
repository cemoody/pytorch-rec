import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.variational_biased_embedding import VariationalBiasedEmbedding

# Exactly the same as the MFClassic model, but arguably
# cleaner code


class MFDeep1(nn.Module):
    is_train = True

    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1., loss=nn.MSELoss):
        super(MFDeep1, self).__init__()
        self.embed_user = VariationalBiasedEmbedding(n_users, n_dim, lb=lub,
                                                     lv=luv, n_obs=n_obs)
        self.embed_item = VariationalBiasedEmbedding(n_items, n_dim, lb=lib,
                                                     lv=liv, n_obs=n_obs)
        self.lin1 = nn.Linear(n_dim * 3, n_dim)
        self.lin2 = nn.Linear(n_dim, 1)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        self.n_obs = n_obs
        self.lossf = loss()

    def forward(self, u, i):
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        x0 = torch.cat((vu, vi, vu * vi), dim=1)
        x1 = F.dropout(self.lin1(x0), self.is_train)
        x2 = self.lin2(x1).squeeze()
        logodds = bias + bi + bu + x2
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = self.lossf(prediction, target)
        reg = (self.embed_user.prior() + self.embed_item.prior()) / n_batches
        return llh + reg
