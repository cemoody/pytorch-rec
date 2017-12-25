import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.mfdeep1 import MFDeep1


class Reverse(torch.autograd.Function):
    def forward(self, input):
        return input

    def backward(self, grad_output):
        return -grad_output


def shuffle(v):
    x = v.data
    s = x[torch.randperm(x.size()[0])]
    return Variable(s)


def is_finite(x):
    return (x * 1.5 != x).data[0]


class MFGAN(nn.Module):
    run_disc = False

    def __init__(self, *args, **kwargs):
        super(MFGAN, self).__init__()
        self.gen = MFDeep1(*args, **kwargs)
        self.dsc = MFDeep1(*args, **kwargs)

    def forward(self, ru, ri):
        self.ru, self.ri = ru, ri
        rp = self.gen.forward(ru, ri)
        return rp

    def disc(self, real_input, real_output, fake_input, fake_output):
        real_pred = self.dsc.forward(*real_input)
        fake_pred = self.dsc.forward(*fake_input)
        real_diff = real_output - real_pred
        fake_diff = fake_output - fake_pred
        # To discriminate, we'll simply compute which is farthest
        # from our prediction. If the difference in real is small
        # compare to fake diff we'll choose real
        loss = F.logsigmoid(real_diff - fake_diff)
        sclr = loss.sum()
        assert is_finite(sclr)
        return sclr

    def loss(self, rp, target):
        # Maximize llh of generator on real data
        loss_gen = self.gen.loss(rp, target)
        assert (loss_gen == loss_gen).data[0]
        assert is_finite(loss_gen)

        if self.run_disc:
            # Run fake data through generator
            ru, ri = self.ru, self.ri
            fu, fi = shuffle(ru), shuffle(ri)
            fp = self.gen.forward(fu, fi)
            # Reverse the gradient; this makes the discriminator better
            # at fooling the generator and improves the generator to
            # fool the discriminator
            _fp = Reverse()(fp)
            loss_disc = self.disc((ru, ri), rp, (fu, fi), _fp)
            assert (loss_disc == loss_disc).data[0]
            assert is_finite(loss_disc)
            return loss_gen, loss_disc
        else:
            return loss_gen
