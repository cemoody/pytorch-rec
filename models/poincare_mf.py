import torch
from torch import nn
from torch.nn import Parameter

from models.poincare import distance
from models.poincare import distance_simple
from models.poincare import project_disc
from models.poincare import transform
from models.poincare import l2

from models.variational_embedding \
        import VariationalEmbedding as VE

from models.biased_embedding import BiasedEmbedding as BE

class PoincareMF(nn.Module):
    def __init__(self, n_users, n_items, n_dim, n_obs, alpha=1e3):
        super(PoincareMF, self).__init__()
        # self.embed_user = nn.Embedding(n_users, n_dim)
        # self.embed_item = nn.Embedding(n_items, n_dim)
        self.embed_user = VE(n_users, n_dim).double()
        self.embed_item = VE(n_items, n_dim).double()
        self.embed_user_bias = VE(n_users, 1).double()
        self.embed_item_bias = VE(n_items, 1).double()
        # self.embed_user.bias.weight.data.uniform_(-1e-3, 1e-3)
        # self.embed_item.bias.weight.data.uniform_(-1e-3, 1e-3)
        # self.embed_user.vect.weight.data.uniform_(-1e-3, 1e-3)
        # self.embed_item.vect.weight.data.uniform_(-1e-3, 1e-3)
        # self.embed_user.vect_mu.weight.data.uniform_(-1e-3, 1e-3)
        # self.embed_item.vect_mu.weight.data.uniform_(-1e-3, 1e-3)
        # self.embed_user.vect_mu = self.embed_user.vect_mu.double()
        # self.embed_user.vect_lv = self.embed_user.vect_lv.double()
        # self.embed_item.vect_mu = self.embed_item.vect_mu.double()
        # self.embed_item.vect_lv = self.embed_item.vect_lv.double()
        # self.embed_user.vect_lv.weight.data.uniform_(-6, -2)
        # self.embed_item.vect_lv.weight.data.uniform_(-6, -2)
        self.offset = Parameter(torch.ones(1).double())
        self.lin = nn.Linear(1, 1).double()
        # self.lin.weight.data = self.lin.weight.data.double()
        # self.lin.bias = self.lin.bias.double()
        self.alpha = alpha
        self.n_obs = n_obs

    def forward(self, u, i):
        u, i = u.squeeze(), i.squeeze()
        vu = self.embed_user(u).double()
        vi = self.embed_item(i).double()
        bu = self.embed_user_bias(u).double().squeeze()
        bi = self.embed_item_bias(i).double().squeeze()
        offset = self.offset.expand(len(u)).double()
        # Force incoming vector to have radius <= 1
        pu, pi = transform(transform(vu)), transform(transform(vi))
        assert l2(pi).max().data[0] < 1.0
        assert l2(pu).max().data[0] < 1.0
        # pu, pi = project_disc(vu), project_disc(vi)
        # pu, pi = vu, vi
        dist = distance_simple(pu, pi)
        # rad = l2(pu) - l2(pi)
        # This penalizes when user is lower than item in the hierarchy
        # e.g. when pu has a higher norm than pi
        # score = (1 + self.alpha * rad) * dist
        score = self.lin((bu + bi + offset + dist).unsqueeze(1)).squeeze()
        assert torch.sum(score != score).data[0] == 0
        return score

    def loss(self, prediction, target):
        # when user u has liked item i then target +1 
        # and score is high but loss negative
        # when user u has disliked item i then target -1
        # loss = -prediction * target
        lossf = torch.nn.MSELoss()
        llh = lossf(prediction, target.double())
        # n_batches = self.n_obs * 1.0 / target.size()[0]
        reg = (self.embed_user.prior() + self.embed_item.prior() +
               self.embed_user_bias.prior() + self.embed_item_bias.prior()
              ) / self.n_obs
        # import pdb; pdb.set_trace()
        return llh + reg
