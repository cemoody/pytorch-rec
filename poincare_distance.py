import torch
from torch.autograd import Function


def arccosh(x):
    # log(x + sqrt(x^2 -1)
    # log(x (1 + sqrt(x^2 -1)/x))
    # log(x) + log(1 + sqrt(x^2 -1)/x)
    # log(x) + log(1 + sqrt((x^2 -1)/x^2))
    # log(x) + log(1 + sqrt(1 - x^-2))
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    return c0 + c1


def l2(x):
    return dot(x, x)


def dot(a, b):
    return (a * b).sum(dim=1)


class PoincareDistance(Function):

    def forward(self, t, x):
        norm = l2(t - x)
        alpha = 1. - l2(t)
        beta = 1. - l2(x)
        gamma = 1 + 2 * norm / alpha / beta
        ret = arccosh(gamma)
        self.save_for_backward(t, x, alpha, beta, gamma)
        return ret

    def ddistdt(sefl, t, x, alpha, beta, gamma):
        c0 = 4 / beta * torch.rsqrt(gamma * gamma - 1)
        c1 = l2(x) - 2 * dot(t, x) + 1
        c2 = alpha * alpha
        return c0 * (c1 / c2 * t - x / alpha)

    def backward(self, grad_output):
        # Return gradient for u & v
        t, x, alpha, beta, gamma = self.saved_tensors
        dddt = self.ddistdt(t, x, alpha, beta, gamma)
        dddx = self.ddistdt(x, t, alpha, beta, gamma)
        grad_t = alpha * alpha * dddt
        grad_x = alpha * alpha * dddx
        return grad_t, grad_x


def poincare_distance(u, v):
    return PoincareDistance()(u, v)
