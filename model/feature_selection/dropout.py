import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model.feature_selection.interface import IFeatureSelector

_k1 = 0.63576
_k2 = 1.87320
_k3 = 1.48695


class VariationalDropoutSelector(IFeatureSelector):

    def __init__(self, channels: int, dim: int = 2, alpha: float = 0.99, adaptive_weight=False):
        super(VariationalDropoutSelector, self).__init__()
        self.channels = channels
        self.dim = dim
        self.alpha = alpha
        self.register_parameter('_log_betas', nn.Parameter(torch.zeros(channels), requires_grad=True))
        self.register_parameter('_log_var', nn.Parameter(torch.zeros(channels), requires_grad=True))
        self.register_parameter('_log_weight', nn.Parameter(torch.zeros(1), requires_grad=True) if adaptive_weight else None)

    @property
    def betas(self):
        return torch.exp(self._log_betas)

    def forward(self, x: Tensor) -> Tensor:
        shape = [1 if i != self.dim else self.channels for i in range(len(x.shape))]
        w = self.betas.view(shape)
        w = w + torch.exp(self._log_var / 2.).view(shape) * torch.randn_like(x)
        return x * w

    def compute_penalty(self, weight=1.):
        log_alphas = self._log_var - 2. * self._log_betas
        t1 = _k1 * torch.sigmoid(_k2 + _k3 * log_alphas)
        t2 = 0.5 * F.softplus(-log_alphas, beta=1.)
        weight = weight * self.compute_adaptive_weight()
        return weight * torch.sum(- t1 + t2 + _k1)

    def compute_adaptive_weight(self):
        try:
            ada_weight = self._log_weight
        except AttributeError:
            ada_weight = None
        if ada_weight is not None:
            return torch.exp(ada_weight)
        return 1.

    @property
    def alphas(self):
        return torch.exp(self._log_var - 2 * self._log_betas)

    @property
    def rates(self):
        alphas = self.alphas
        return alphas / (1. + alphas)

    def get_valid_betas(self):
        rates = self.rates
        return self.betas[rates < self.alpha]

    def get_valid_fraction(self):
        return torch.mean((self.rates < self.alpha).to(torch.float)).item()
