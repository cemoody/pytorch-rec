import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal


def reparameterize(mu, logvar):
    # From VAE example
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    std = logvar.mul(0.5).exp_()
    eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    z = eps.mul(std).add_(mu)
    return z


class VariationalBiasedEmbedding(nn.Module):
    def __init__(self, n_feat, n_dim, n_obs=1.0, lv=1.0, lb=1.0):
        super(VariationalBiasedEmbedding, self).__init__()
        self.vect_mu = nn.Embedding(n_feat, n_dim)
        self.vect_lv = nn.Embedding(n_feat, n_dim)
        self.bias = nn.Embedding(n_feat, 1)
        self.n_obs = n_obs
        self.lv = lv
        self.lb = lb
        xavier_normal(self.vect_mu.weight)
        xavier_normal(self.vect_lv.weight)
        xavier_normal(self.bias.weight)
        self.vect_lv.weight.data -= 3

    def __call__(self, index):
        vec = reparameterize(self.vect_mu(index), self.vect_lv(index))
        return self.bias(index).squeeze(), vec

    def prior(self):
        mu = self.vect_mu.weight
        lv = self.vect_lv.weight
        kld = torch.sum(0.5 * (mu**2 + torch.exp(lv) - lv - 1))
        l2 = (self.bias.weight**2.0).sum()
        loss = (l2 * self.lb + kld * self.lv)
        return loss
