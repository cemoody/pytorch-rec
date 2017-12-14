import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from models.variational_biased_embedding import VariationalBiasedEmbedding


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class MFDeep1(nn.Module):
    is_train = True

    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1., loss=nn.MSELoss(size_average=True)):
        super(MFDeep1, self).__init__()
        self.embed_user = VariationalBiasedEmbedding(n_users, n_dim, lb=lub,
                                                     lv=luv, n_obs=n_obs)
        self.embed_item = VariationalBiasedEmbedding(n_items, n_dim, lb=lib,
                                                     lv=liv, n_obs=n_obs)
        self.lin1 = nn.Linear(n_dim, n_dim, bias=True)
        self.lin2 = nn.Linear(n_dim, n_dim, bias=True)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        self.n_obs = n_obs
        self.lossf = loss
        self.lin1.weight.data
        self.glob_bias.data[:] = 1e-6

    def forward(self, u, i):
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        x1 = self.lin1(vu) * self.lin2(vi)
        logodds = bias + bi + bu + x1.sum(dim=1)  # + x0.sum(dim=1)
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = self.lossf(prediction, target)
        mat1 = self.lin1.weight @ torch.t(self.lin1.weight)
        mat2 = self.lin2.weight @ torch.t(self.lin2.weight)
        eye1 = Variable(torch.eye(*mat1.size()))
        eye2 = Variable(torch.eye(*mat2.size()))
        diff1 = ((mat1 - eye1)**2.0).sum()
        diff2 = ((mat2 - eye2)**2.0).sum()
        reg = (diff1 + diff2 + self.embed_user.prior() +
               self.embed_item.prior())
        return llh + reg / n_batches
