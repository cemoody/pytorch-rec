from torch import nn
from torch.nn.init import xavier_normal


class BiasedEmbedding(nn.Module):
    def __init__(self, n_feat, n_dim, lv=1.0, lb=1.0):
        super(BiasedEmbedding, self).__init__()
        self.vect = nn.Embedding(n_feat, n_dim)
        self.bias = nn.Embedding(n_feat, 1)
        self.lv = lv
        self.lb = lb
        xavier_normal(self.vect.weight)
        xavier_normal(self.bias.weight)

    def __call__(self, index):
        return self.bias(index).squeeze(), self.vect(index)

    def prior(self):
        loss = (self.vect.weight.sum()**2.0 * self.lv +
                self.bias.weight.sum()**2.0 * self.lb)
        return loss
