import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from models.biased_embedding import BiasedEmbedding


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
        self.embed_feat = BiasedEmbedding(n_features, n_dim, lb=lb, lv=lv)
        self.glob_bias = Parameter(torch.FloatTensor([0.01]))
        self.n_obs = n_obs
        self.lb = lb
        self.lv = lv
        self.lossf = loss()

    def forward(self, input):
        idx = input
        biases = index_into(self.embed_feat.bias.weight, idx).squeeze()
        vectrs = index_into(self.embed_feat.vect.weight, idx)
        vector = factorization_machine(vectrs).squeeze()
        logodds = biases.sum(dim=1) + vector.sum(dim=1)
        return logodds

    def loss(self, prediction, target):
        n_batches = self.n_obs * 1.0 / target.size()[0]
        llh = self.lossf(prediction, target)
        reg = self.embed_feat.prior() / n_batches
        return llh + reg
