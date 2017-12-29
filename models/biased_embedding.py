import torch
from torch import nn
from torch.nn import Parameter


class BiasedEmbedding(nn.Module):
    def __init__(self, n_feat, n_dim, lv=1.0, lb=1.0):
        super(BiasedEmbedding, self).__init__()
        self.vect = nn.Embedding(n_feat, n_dim)
        self.bias = nn.Embedding(n_feat, 1)
        self.off_vect = Parameter(torch.zeros(1, n_dim))
        self.mul_vect = Parameter(torch.ones(1, n_dim))
        self.off_bias = Parameter(torch.zeros(1))
        self.mul_bias = Parameter(torch.ones(1))
        self.n_dim = n_dim
        self.n_feat = n_feat
        self.lv = lv
        self.lb = lb
        self.vect.weight.data.normal_(0, 1.0 / n_dim)
        self.bias.weight.data.normal_(0, 1.0 / n_dim)

    def __call__(self, index):
        assert (index.max() < self.n_feat).all()
        assert (index.min() >= 0).all()
        off_vect = self.off_vect.expand(len(index), self.n_dim).squeeze()
        off_bias = self.off_bias.expand(len(index), 1).squeeze()
        bias = off_bias + self.mul_bias * self.bias(index).squeeze()
        vect = off_vect + self.mul_vect * self.vect(index)
        return bias, vect

    def prior(self):
        loss = ((self.vect.weight**2.0).sum() * self.lv +
                (self.bias.weight**2.0).sum() * self.lb)
        return loss
