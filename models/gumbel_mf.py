import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from models.gumbel import gumbel_softmax_correlated
from models.gumbel import gumbel_softmax
from models.gumbel import hellinger
from models.biased_embedding import BiasedEmbedding


class GumbelMF(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1., tau=0.8, loss=nn.MSELoss):
        super(GumbelMF, self).__init__()
        self.embed_user = BiasedEmbedding(n_users, n_dim, lb=lub, lv=luv)
        self.embed_item = BiasedEmbedding(n_items, n_dim, lb=lib, lv=liv)
        self.glob_bias = Parameter(torch.FloatTensor([0.01]))
        self.n_obs = n_obs
        self.lossf = loss()
        self.tau = tau

    def forward(self, u, i):
        u, i = u.squeeze(), i.squeeze()
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, lu = self.embed_user(u)
        bi, li = self.embed_item(i)
        if self.training:
            du, di = gumbel_softmax_correlated(lu, li)
            # du = gumbel_softmax(lu, self.tau)
            # di = gumbel_softmax(li, self.tau)
        else:
            du = F.softmax(lu)
            di = F.softmax(li)
        intx = hellinger(du, di)
        logodds = (bias + bi + bu + intx).squeeze()
        return logodds

    def loss(self, prediction, target):
        # average likelihood loss per example
        ex_llh = self.lossf(prediction, target)
        # regularization penalty summed over whole model
        epoch_reg = (self.embed_user.prior() + self.embed_item.prior())
        # penalty should be computer for a single example
        ex_reg = epoch_reg * 1.0 / self.n_obs
        return ex_llh + ex_reg
