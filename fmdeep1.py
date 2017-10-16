import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layernorm import LayerNorm
from fm import index_into
from fm import factorization_machine

# http://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf
# https://www.oreilly.com/ideas/deep-matrix-factorization-using-apache-mxnet
# A 'deep' FM model, with non-linearities after the FM


class FMDeep1(nn.Module):
    def __init__(self, n_features, n_dim, n_obs,
                 lb=1., lv=1.):
        super(FMDeep1, self).__init__()
        self.feat_bias = nn.Embedding(n_features, 1)
        self.feat_vect = nn.Embedding(n_features, n_dim)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        self.layer_norm1 = LayerNorm(n_dim)
        self.layer_norm2 = LayerNorm(n_dim)
        self.lin1 = nn.Linear(n_dim, n_dim)
        self.lin2 = nn.Linear(n_dim, n_dim)
        self.n_obs = n_obs

    def forward(self, idx):
        biases = index_into(self.feat_bias.weight, idx).squeeze()
        vectrs = index_into(self.feat_vect.weight, idx)
        x0 = factorization_machine(vectrs).squeeze()
        x1 = x0 + self.lin1(self.layer_norm1(x0))
        x2 = x1 + self.lin2(self.layer_norm2(x1))
        logodds = biases.sum(dim=1) + x2.sum(dim=1)
        return logodds

    def prior(self):
        loss = ((self.feat_bias.weight**2.0).sum() * self.lb
                (self.feat_vect.weight**2.0).sum() * self.lv)
        return loss

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = F.binary_cross_entropy_with_logits(prediction, target)
        reg = self.prior() / n_batches
        return llh + reg
