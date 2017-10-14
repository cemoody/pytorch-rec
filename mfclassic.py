import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# A simple matrix factorization model
# where y is the binary outcome,
# u is the user index i is the item index
# and M and W are user and item matrices
# then
# y_ui ~ a + b_u + c_i + (M_u . W_i)
# Which says that a user vector (M_u) interacts with an
# item vector (W_i) via a dot product
# plus a user bias (b_u) and an item bias (c_i)
# and a global bias (a)
# and so the total loss is:
# L = logistic(y_ui, prediction) + regularization


class MFClassic(nn.Module):
    def __init__(self, n_users, n_items, dim, n_obs, reg_user_bas=1.,
                 reg_item_bas=1., reg_user_vec=1., reg_item_vec=1.):
        super(MFClassic, self).__init__()
        # User & item biases
        self.glob_bas = Parameter(torch.Tensor(1, 1))
        self.user_bas = nn.Embedding(n_users, 1)
        self.item_bas = nn.Embedding(n_items, 1)
        # User & item vectors
        self.user_vec = nn.Embedding(n_users, dim)
        self.item_vec = nn.Embedding(n_items, dim)
        self.n_obs = n_obs
        self.reg_user_bas = reg_user_bas
        self.reg_user_vec = reg_user_vec
        self.reg_item_bas = reg_item_bas
        self.reg_item_vec = reg_item_vec

    def forward(self, user_idx, item_idx):
        batchsize = len(user_idx)
        glob_bas = self.glob_bas.expand(batchsize, 1).squeeze()
        user_bas = self.user_bas(user_idx).squeeze()
        item_bas = self.item_bas(item_idx).squeeze()
        user_vec = self.user_vec(user_idx)
        item_vec = self.item_vec(item_idx)
        intx = (user_vec * item_vec).sum(dim=1)
        score = glob_bas + user_bas + item_bas + intx
        return score

    def loss(self, prediction, target):
        # Measure likelihood of target rating given prediction
        # Same as a logistic / sigmoid loss -- it compares a score
        # that ranges from -inf to +inf with a binary outcome of 0 or 1
        llh = F.binary_cross_entropy_with_logits(prediction, target)
        # L2 regularization of weights with custom coefficients
        # for each piece
        prior = (self.user_bas.weight.sum()**2. * self.reg_user_bas +
                 self.item_bas.weight.sum()**2. * self.reg_item_bas +
                 self.user_vec.weight.sum()**2. * self.reg_user_vec +
                 self.item_vec.weight.sum()**2. * self.reg_item_vec)
        # Since we're computing in minibatches but the prior is computed
        # once over a single pass thru dataset, adjust the prior loss s.t.
        # it's in proportion to the number of minibatches. This is degenerate
        # with the regularization coefficients, so not necessary, but it
        # means our initial gueeses for regularization coefs are around 1.0
        n_minibatches = self.n_obs * 1.0 / target.size()[0]
        prior_weighted = prior / n_minibatches
        return llh + prior_weighted
