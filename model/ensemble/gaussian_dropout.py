import math

import torch
from torch import nn, Tensor


class GaussianDropout(nn.Module):

    def __init__(self, p: float=0.5):
        super(GaussianDropout, self).__init__()
        self.p = p

    @property
    def std(self):
        return math.sqrt(self.p / (1. - self.p))

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0.:
            sample = torch.normal(1., self.std, x.shape, device=x.device)
            return x * sample
        return x

    def extra_repr(self) -> str:
        return f'p={self.p}'
