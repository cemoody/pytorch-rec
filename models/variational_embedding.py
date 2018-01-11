import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal


def reparameterize(mu, logvar):
    # From VAE example
    # https://github.com/pytorch/examples/blob/master/vae/main.py
    std = logvar.mul(0.5).exp_()
    typ = type(mu.data)
    eps = typ(std.size()).normal_()
    eps = Variable(eps)
    z = eps.mul(std).add_(mu)
    return z


def index_into(arr, idx):
    new_shape = (idx.size()[0], idx.size()[1], arr.size()[1])
    return arr[idx.resize(torch.numel(idx.data))].view(new_shape)


class VariationalEmbedding(nn.Module):
    def __init__(self, n_feat, n_dim):
        super(VariationalEmbedding, self).__init__()
        self.vect_mu = nn.Embedding(n_feat, n_dim)
        self.vect_lv = nn.Embedding(n_feat, n_dim)
        xavier_normal(self.vect_mu.weight)
        xavier_normal(self.vect_lv.weight)

    def __call__(self, index=None):
        if index is None:
            mu = self.vect_mu.weight
            lv = self.vect_lv.weight
        elif len(index.data.size()) > 1:
            mu = index_into(self.vect_mu.weight, index)
            lv = index_into(self.vect_lv.weight, index)
        else:
            mu = self.vect_mu(index)
            lv = self.vect_lv(index)
        vec = reparameterize(mu, lv)
        return vec

    def prior(self, mu=None, lv=None):
        if mu is None and lv is None:
            return self.prior_zero_one()
        else:
            return self.prior_kldiv(mu, lv)

    def prior_zero_one(self):
        mu = self.vect_mu.weight
        lv = self.vect_lv.weight
        kld = torch.sum(0.5 * (mu**2 + torch.exp(lv**2.0) - lv - 1))
        return kld

    def prior_kldiv(self, mu2, lv2):
        # Full KL divergence between two univariate Gaussians
        mu1 = self.vect_mu.weight
        lv1 = self.vect_lv.weight
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        kld = (lv2 - lv1 +
               0.5 * torch.exp(lv1 - lv2)**2.0 +
               # 0.5 * torch.exp(lv1 * (1 - lv2 / lv1))**2.0 +
               # 0.5 * torch.exp(lv1)**2.0 * torch.exp(1 - lv2 / lv1))**2.0 +
               # 0.5 * v1**2.0 * torch.exp(1 - lv2 / lv1)**2.0 +
               # 0.5 * v1**2.0 * torch.exp(1 - lv2 / lv1)**2.0 +
               0.5 * ((mu1 - mu2)**2.0)/(v2**2.0) -
               0.5)
        return kld.sum()
