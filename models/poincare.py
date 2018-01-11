import torch
from torch.autograd import Function
import torch.nn.functional as F


def arccosh(x):
    # log(x + sqrt(x^2 -1)
    # log(x (1 + sqrt(x^2 -1)/x))
    # log(x) + log(1 + sqrt(x^2 -1)/x)
    # log(x) + log(1 + sqrt((x^2 -1)/x^2))
    # log(x) + log(1 + sqrt(1 - x^-2))
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    return c0 + c1


def arccosh1p(x):
    # same as arccos(1 + x)
    c0 = torch.log1p(x)
    x1p = x + 1
    c1 = torch.log1p(torch.sqrt(x1p * x1p - 1) / x1p)
    return c0 + c1


def norm(x):
    return torch.sqrt(l2(x))


def l2(x):
    return dot(x, x)


def dot(a, b):
    return (a * b).sum(dim=1)


def ddistdt(t, x, alpha, beta, gamma, eps=1e-6):
    c0 = (4 / beta * torch.rsqrt(gamma * gamma - 1 + eps))
    c1 = (l2(x) - 2 * dot(t, x) + 1).unsqueeze(1)
    c2 = (alpha * alpha)
    return c0 * (c1 / (c2 + eps) * t - x / (alpha + eps))


class PoincareDistance(Function):

    def forward(self, t, x):
        norm = l2(t - x)
        alpha = 1. - l2(t)
        beta = 1. - l2(x)
        # gamma = 1 + 2 * norm / alpha / beta
        gamma1m = 2 * norm / alpha / beta
        gamma = gamma1m + 1
        # ret = arccosh(gamma)
        ret = arccosh1p(gamma1m)
        self.save_for_backward(t, x)
        self.saved_args = (alpha, beta, gamma)
        return ret
    
    def backward(self, grad_output):
        # Return gradient for u & v
        t, x = self.saved_tensors
        alpha, beta, gamma = [a.unsqueeze(1) for a in self.saved_args]
        dddt = ddistdt(t, x, alpha, beta, gamma)
        dddx = ddistdt(x, t, alpha, beta, gamma)
        grad_t = alpha * alpha * dddt
        grad_x = alpha * alpha * dddx
        return grad_t, grad_x


def distance(u, v):
    return PoincareDistance()(u, v)


def clamp(theta, eps=1e-5):
    norm = torch.sqrt(l2(theta)).unsqueeze(1)
    flag = (norm >= 1.0).expand_as(theta)
    theta[flag] = (theta / norm - eps)[flag]
    return theta


class ProjectDisc(Function):

    def forward(self, theta):
        self.save_for_backward(theta)
        return clamp(theta)
    
    def backward(self, grad_output):
        # Return gradient for theta
        theta, = self.saved_tensors
        coeff = (1 - l2(theta))**2.0 / 4
        grad_theta = coeff * grad_output
        return grad_theta


def project_disc(theta):
    return ProjectDisc()(theta)


def transform(theta, eps=1e-9):
    # as l2t-> 0 norm->theta, div = 1
    # as l2t-> 1 norm->theta / ||l2t||, div = ||theta||
    # as l2t-> +inf norm->theta / ||l2t||, div = ||theta||
    # when l2t = 0: div = (1 + ( 0) (+1)) = 1
    # when l2t = 1: div = (1 + ( 0) (+1)) = 1
    # when l2t = 2: div = (1 + ( 1) (+1)) = 2
    # when l2t = 3: div = (1 + ( 2) (+1)) = 3
    # div = (1 + relu(l2t - 1))
    # or with ELU:
    # relu(x) = elu(x - 1) + 1
    # div = 1 + (1 + elu(l2t - 2))
    # div = 2 + elu(l2t - 2)
    # when l2t = 0: div = (2 + (-1.0)) = 1
    # when l2t = 1: div = (2 + (-0.3)) = 1.7
    # when l2t = 2: div = (2 + ( 0.0)) = 2
    # when l2t = 3: div = (2 + (+1.0)) = 3
    # theta = torch.clamp(theta, -cmax, cmax)
    l2t = torch.sqrt(l2(theta)).unsqueeze(1) + eps
    div = 2 + F.elu(l2t - 2)
    ret = theta / (div + eps)
    # l2t = torch.sqrt(l2(ret)).unsqueeze(1)
    # ret2 = ret / (F.relu(l2t - 1) + 1 + eps)
    return ret


def distance_simple(u, v, eps=1e-9):
    diff = norm(u - v)
    alpha = torch.sqrt(1. - l2(u))
    beta = torch.sqrt(1. - l2(v))
    root = l2(u) * l2(v) - 2 * (u * v).sum(dim=1) + 1
    num = diff + torch.sqrt(root)
    div = alpha * beta
    ret = 2 * torch.log(num / div)
    return ret
