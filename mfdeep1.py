import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layernorm import LayerNorm
from biased_embedding import BiasedEmbedding

# Exactly the same as the MFClassic model, but arguably
# cleaner code


class MFDeep1(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, lub=1.,
                 lib=1., luv=1., liv=1.):
        super(MFDeep1, self).__init__()
        self.embed_user = BiasedEmbedding(n_users, n_dim, lb=lub, lv=luv)
        self.embed_item = BiasedEmbedding(n_items, n_dim, lb=lib, lv=liv)
        self.layer_norm0 = LayerNorm(n_dim)
        self.layer_norm1 = LayerNorm(n_dim)
        self.layer_norm2 = LayerNorm(n_dim)
        self.lin1 = nn.Linear(n_dim, n_dim)
        self.lin2 = nn.Linear(n_dim, 1)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        self.n_obs = n_obs

    def forward(self, u, i):
        bias = self.glob_bias.expand(len(u), 1).squeeze()
        bu, vu = self.embed_user(u)
        bi, vi = self.embed_item(i)
        x0 = (vu * vi)
        x1 = self.lin1(x0)
        x2 = self.lin2(x1).squeeze()
        logodds = bias + bi + bu + x2
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = F.binary_cross_entropy_with_logits(prediction, target)
        reg = (self.embed_user.prior() + self.embed_item.prior()) / n_batches
        return llh + reg
