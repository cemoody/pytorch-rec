import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# A simple matrix factorization model with a 2nd order polynomial on age


class MFPoly2(nn.Module):
    def __init__(self, n_users, n_items, dim, n_obs, reg_user_bas=1.,
                 reg_item_bas=1., reg_user_vec=1., reg_item_vec=1.):
        super(MFPoly2, self).__init__()
        self.glob_bas = Parameter(torch.Tensor(1, 1))
        self.user_bas = nn.Embedding(n_users, 1)
        self.item_bas = nn.Embedding(n_items, 1)
        self.user_vec = nn.Embedding(n_users, dim)
        self.item_vec = nn.Embedding(n_items, dim)
        # 2nd order polynomial = pass through 2 linear layers
        # each linear layer applies y = A x + b
        # age is 1D and output is 1D, so transformattion
        # matrix A has shape (1, 1)
        self.age1 = nn.Linear(1, 1)
        self.age2 = nn.Linear(1, 1)
        self.n_obs = n_obs
        self.reg_user_bas = reg_user_bas
        self.reg_user_vec = reg_user_vec
        self.reg_item_bas = reg_item_bas
        self.reg_item_vec = reg_item_vec

    def forward(self, user_idx, item_idx, age):
        batchsize = len(user_idx)
        glob_bas = self.glob_bas.expand(batchsize, 1).squeeze()
        user_bas = self.user_bas(user_idx).squeeze()
        item_bas = self.item_bas(item_idx).squeeze()
        user_vec = self.user_vec(user_idx)
        item_vec = self.item_vec(item_idx)
        intx = (user_vec * item_vec).sum(dim=1)
        age_effect = self.age2(self.age1(age.view(batchsize, 1))).squeeze()
        score = glob_bas + user_bas + item_bas + intx + age_effect
        import pdb; pdb.set_trace()
        return score

    def loss(self, prediction, target):
        llh = F.binary_cross_entropy_with_logits(prediction, target)
        prior = (self.user_bas.weight.sum()**2. * self.reg_user_bas +
                 self.item_bas.weight.sum()**2. * self.reg_item_bas +
                 self.user_vec.weight.sum()**2. * self.reg_user_vec +
                 self.item_vec.weight.sum()**2. * self.reg_item_vec)
        n_minibatches = self.n_obs * 1.0 / target.size()[0]
        prior_weighted = prior / n_minibatches
        return llh + prior_weighted
