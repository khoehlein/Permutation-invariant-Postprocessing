from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class IQuantileIntegrator(nn.Module):

    def __init__(self, degree: int, dim: int = 1):
        super(IQuantileIntegrator, self).__init__()
        self.degree = degree
        self.dim = dim
        nodes, weights, scale = self.compute_parameters()
        self._assert_expected_shape(nodes)
        if weights is not None:
            self._assert_expected_shape(weights)
        self.register_buffer('nodes', self._float32_or_none(nodes))
        self.register_buffer('weights', self._float32_or_none(weights))
        self.global_scale = scale

    @staticmethod
    def _float32_or_none(x: Union[Tensor, None]):
        if x is not None:
            return x.to(torch.float32)
        return x

    def _assert_expected_shape(self, x: Tensor):
        assert len(x.shape) == 1 and len(x) == self.degree, \
            f'[ERROR] Tensor of shape {x.shape} does not match the shape requirements of quantile integrator with degree {self.degree}.'

    def compute_parameters(self) -> Tuple[Tensor, Union[Tensor, None], Union[float, None]]:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.dim] == self.degree
        if self.weights is not None:
            shape = [1 for _ in x.shape]
            shape[self.dim] = self.degree
            x = x * self.weights.view(shape)
        if self.global_scale is not None:
            x = x * self.global_scale
        return torch.mean(x, dim=self.dim, keepdim=True)


class UniformQuantileIntegrator(IQuantileIntegrator):

    def compute_parameters(self) -> Tuple[Tensor, Union[Tensor, None], Union[float, None]]:
        nodes = torch.arange(1, self.degree + 1) / (self.degree + 1)
        return nodes, None, None


class MidpointQuantileIntegrator(IQuantileIntegrator):

    def compute_parameters(self) -> Tuple[Tensor, Union[Tensor, None], Union[float, None]]:
        nodes = (torch.arange(self.degree) + 0.5) / self.degree
        return nodes, None, None


class GaussLegendreQuantileIntegrator(IQuantileIntegrator):

    def compute_parameters(self) -> Tuple[Tensor, Union[Tensor, None], Union[float, None]]:
        nodes, weights = np.polynomial.legendre.leggauss(self.degree)
        nodes = (torch.from_numpy(nodes) + 1.) / 2.
        weights = (self.degree / 2.) * torch.from_numpy(weights)
        return nodes, weights, None