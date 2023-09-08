import torch
from torch import nn, Tensor

from model.feature_selection.interface import VarianceTracker, IFeatureSelector


class SmallifySelector(IFeatureSelector):
    """
    Smallify Switch layer based on description in Leclerc et al. (2018)
    https://arxiv.org/pdf/1806.03723.pdf
    """

    def __init__(self, channels: int, dim: int = 2, momentum: float = 0.99, alpha: float = 0.99):
        super(SmallifySelector, self).__init__()
        self.channels = channels
        self.dim = dim
        self.register_parameter('betas', nn.Parameter(torch.normal(1., 1.e-4, (channels,)), requires_grad=True))
        self.tracker = VarianceTracker(torch.sign(self.betas).detach(), momentum=momentum, alpha=alpha)

    def forward(self, x: Tensor):
        shape = [self.channels if i == self.dim else 1 for i in range(len(x.shape))]
        if self.training:
            self.tracker.update(torch.sign(self.betas).detach())
            self.betas.view(shape) * x
        return self.get_valid_betas().view(shape) * x

    def get_valid_betas(self):
        is_valid = self.tracker.get_valid_positions()
        out = torch.zeros_like(self.betas)
        out[is_valid] = self.betas[is_valid]
        return out

    def compute_penalty(self, weight=1.):
        return torch.mean(torch.abs(self.betas)) * weight

    def get_valid_fraction(self):
        return torch.mean(self.tracker.get_valid_positions().to(torch.float)).item()
