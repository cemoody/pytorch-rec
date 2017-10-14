import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from poincare_distance import poincare_distance


class BiasedEmbedding(nn.Module):
    def __init__(self, n_feat, n_dim):
        super(BiasedEmbedding, self).__init__()
        self.vect = nn.Embedding(n_feat, n_dim)
        self.bias = nn.Embedding(n_feat, 1)

    def __call__(self, index):
        return self.bias(index).squeeze(), self.vect(index)


class MFPoincare(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1.):
        super(MFPoincare, self).__init__()
        self.embed_user = BiasedEmbedding(n_users, n_dim, lb=lub, lv=luv)
        self.embed_item = BiasedEmbedding(n_items, n_dim, lb=lib, lv=liv)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        self.n_obs = n_obs

    def forward(self, u, i):
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        dist = poincare_distance(vu, vi)
        logodds = bias + bi + bu + dist
        return logodds

    def loss(self, prediction, target):
        # Don't know how to regularize poincare space yet!
        llh = F.binary_cross_entropy_with_logits(prediction, target)
        return llh
