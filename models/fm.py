import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable


def index_into(arr, idx):
    new_shape = (idx.size()[0], idx.size()[1], arr.size()[1])
    return arr[idx.resize(torch.numel(idx.data))].view(new_shape)


def factorization_machine(v, w=None):
    # Takes an input 2D matrix v that has of n vectors, each d-dimensional
    # produces output `x` that is d-dimensional
    # v is (batchsize, n_features, dim)
    # w is (batchsize, n_features)
    # w functions as a weight array, assumed to be 1 if missing
    # Uses Rendle's trick for computing pairs of features in linear time
    batchsize, n_features, n_dim = v.size()
    if w is None:
        w = Variable(torch.ones(v.size()))
    else:
        w = w.expand(batchsize, n_features, n_dim)
    t0 = (v * w).sum(dim=1)**2.0
    t1 = (v**2.0 * w**2.0).sum(dim=1)
    return 0.5 * (t0 - t1)


class FM(nn.Module):
    def __init__(self, n_features, n_dim, n_obs,
                 lb=1., lv=1., loss=nn.MSELoss):
        super(FM, self).__init__()
        self.feat_bias = nn.Embedding(n_features, 1)
        self.feat_vect = nn.Embedding(n_features, n_dim)
        self.glob_bias = Parameter(torch.Tensor(1, 1))
        self.n_obs = n_obs
        self.lb = lb
        self.lv = lv

    def forward(self, idx):
        biases = index_into(self.feat_bias.weight, idx).squeeze()
        vectrs = index_into(self.feat_vect.weight, idx)
        vector = factorization_machine(vectrs).squeeze()
        logodds = biases.sum(dim=1) + vector.sum(dim=1)
        return logodds

    def prior(self):
        loss = ((self.feat_bias.weight**2.0).sum() * self.lb +
                (self.feat_vect.weight**2.0).sum() * self.lv)
        return loss

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = self.lossf(prediction, target)
        reg = self.prior() / n_batches
        return llh + reg
