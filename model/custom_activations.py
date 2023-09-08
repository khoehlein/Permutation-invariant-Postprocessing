from math import log

import torch
from torch import nn, Tensor


class PSine(nn.Module):

    def __init__(self, omega: float = 1.):
        super(PSine, self).__init__()
        self.register_parameter('log_omega', nn.Parameter(torch.tensor([log(omega)], requires_grad=True)))

    @property
    def omega(self):
        return torch.exp(self.log_omega)

    def forward(self, x: Tensor):
        return torch.sin(self.omega * x)


class Sine(nn.Module):

    def __init__(self, omega: float = 1.):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x: Tensor):
        return torch.sin(self.omega * x)