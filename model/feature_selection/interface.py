import torch
from torch import nn, Tensor


class IFeatureSelector(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def compute_penalty(self, weight=1.):
        raise NotImplementedError()

    def get_valid_fraction(self):
        raise NotImplementedError()

    def get_valid_betas(self):
        raise NotImplementedError()


class VarianceTracker(nn.Module):

    def __init__(self, initial: Tensor, momentum: float = 0.99, alpha: float = 0.99):
        super(VarianceTracker, self).__init__()
        self.momentum = momentum
        self.alpha = alpha
        self.register_buffer('_sign_ema', initial)
        self.register_buffer('_sign_emv', torch.zeros_like(initial))

    def update(self, x: Tensor):
        delta = x - self._sign_ema
        self._sign_ema = self._sign_ema + (1. - self.momentum) * delta
        self._sign_emv = self.momentum * (self._sign_emv + (1. - self.momentum) * delta ** 2)
        return self

    def get_valid_positions(self):
        return self._sign_emv < self.alpha

    @property
    def variances(self):
        return self._sign_emv
