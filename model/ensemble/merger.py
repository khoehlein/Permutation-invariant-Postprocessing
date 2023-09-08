import math
from typing import Optional

import torch
from torch import nn, Tensor

from model.feature_selection.interface import IFeatureSelector


class EnsembleMerger(nn.Module):

    def __init__(self, keepdim=False, dim=-2):
        super(EnsembleMerger, self).__init__()
        self.keepdim = keepdim
        self.dim = dim

    def output_channels(self, in_channels):
        raise NotImplementedError()

    def forward(self, ensemble):
        raise NotImplementedError()

    def extra_repr(self) -> str:
        return 'keepdim={}'.format(self.keepdim)


class MeanMerger(EnsembleMerger):

    def output_channels(self, in_channels):
        return in_channels

    def forward(self, ensemble):
        return torch.mean(ensemble, dim=self.dim, keepdim=self.keepdim)


class MeanStdMerger(EnsembleMerger):

    def __init__(self, unbiased=True, keepdim=False, eps=0., log_std=False, dim=-2):
        super(MeanStdMerger, self).__init__(keepdim=keepdim, dim=dim)
        self.unbiased = unbiased
        self.eps = eps
        self.log_std = log_std

    def output_channels(self, in_channels):
        return 2 * in_channels

    def forward(self, ensemble):
        mu = torch.mean(ensemble, dim=self.dim, keepdim=self.keepdim)
        sigma = torch.std(ensemble, dim=self.dim, keepdim=self.keepdim)
        if self.eps > 0.:
            sigma = sigma + self.eps
        if self.log_std:
            sigma = torch.log(sigma)
        return torch.cat([mu, sigma], dim=[None, -1, -2][self.dim])

    def extra_repr(self) -> str:
        return 'keepdim={}, unbiased={}, eps={}, log_std={}'.format(self.keepdim, self.unbiased, self.eps, self.log_std)


class MaxMerger(EnsembleMerger):

    def output_channels(self, in_channels):
        return in_channels

    def forward(self, ensemble):
        return torch.amax(ensemble, dim=self.dim, keepdim=self.keepdim)


class MinMaxMerger(EnsembleMerger):

    def output_channels(self, in_channels):
        return 2 * in_channels

    def forward(self, ensemble):
        return torch.cat([
            torch.amin(ensemble, dim=self.dim, keepdim=self.keepdim),
            torch.amax(ensemble, dim=self.dim, keepdim=self.keepdim)
        ], dim=[None, -1, -2][self.dim])


class SelectionWrapper(EnsembleMerger):

    def __init__(self, merger: EnsembleMerger, selector: IFeatureSelector):
        super(SelectionWrapper, self).__init__(keepdim=merger.keepdim)
        self.merger = merger
        self.selector = selector

    def _normalize(self, x: Tensor):
        return x / torch.std(x, dim=self.merger.dim, unbiased=True, keepdim=True)

    def output_channels(self, in_channels):
        return self.merger.output_channels(in_channels)

    def forward(self, ensemble):
        return self.selector.forward(self.merger.forward(self._normalize(ensemble)))

    def compute_penalty(self, weight=1.):
        return self.selector.compute_penalty(weight=weight)


class WeightedMeanMerger(EnsembleMerger):

    def __init__(self, channels: int, num_heads: int = 8, keepdim=False, dim=-2):
        super(WeightedMeanMerger, self).__init__(keepdim=keepdim,dim=dim)
        assert channels % num_heads == 0, '[ERROR] Input dimension must be divisible by number of heads.'
        self.channels = channels
        self.num_heads = num_heads
        self.activation = nn.MultiheadAttention(channels, self.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.register_parameter('query', nn.Parameter((torch.rand(1, 1, channels) - 0.5) / 10.))

    def output_channels(self, in_channels: int) -> int:
        assert in_channels == self.channels
        return self.channels

    def forward(self, ensemble):
        out = self.activation(self.query.repeat(len(ensemble), 1, 1), ensemble, ensemble, need_weights=False)[0]
        out = self.norm(self.query + out)
        if not self.keepdim:
            out = out.squeeze(self.dim)
        return out


# class WeightedMeanStdMerger(EnsembleMerger):
#
#     def __init__(self, channels: int, num_heads: int = 8, keepdim=False, dim=-2):
#         super(WeightedMeanStdMerger, self).__init__(keepdim=keepdim,dim=dim)
#         assert channels % num_heads == 0, '[ERROR] Input dimension must be divisible by number of heads.'
#         self.channels = channels
#         self.num_heads = num_heads
#         self.activation = nn.MultiheadAttention(channels, self.num_heads, batch_first=True)
#         self.register_parameter('query', nn.Parameter((torch.rand(1, 1, channels) - 0.5) / 10.))
#
#     def output_channels(self, in_channels: int) -> int:
#         assert in_channels == self.channels
#         return self.channels
#
#     def forward(self, ensemble):
#         mean, weights = self.activation(self.query, ensemble, ensemble)[0]
#         std = torch.sqrt(torch.mean((ensemble - mean), dim=self.dim))
#         if not self.keepdim:
#             out = out.unsqueeze(self.dim)
#         return out


# class WeightedMeanMerger(EnsembleMerger):
#
#     def __init__(self, channels: int, embedding_channels: Optional[int] = None, num_heads: int = 1, keepdim=False, dim=-2):
#         super(WeightedMeanMerger, self).__init__(keepdim=keepdim,dim=dim)
#         if embedding_channels is None:
#             embedding_channels = channels
#         assert channels % num_heads == 0, '[ERROR] Input dimension must be divisible by number of heads.'
#         assert embedding_channels % num_heads == 0, '[ERROR] Embedding dimension must be divisible by number of heads.'
#         self.channels = channels
#         self.embedding_channels = embedding_channels
#         self.num_heads = num_heads
#         self._d = embedding_channels // num_heads
#         self._sqrt_d = math.sqrt(self._d)
#         self.linear = nn.Conv1d(channels, channels + embedding_channels, (1,))
#         self.register_parameter('query', nn.Parameter((torch.rand(num_heads, embedding_channels // num_heads) - 0.5) / 10.))
#
#     def output_channels(self, in_channels: int) -> int:
#         assert in_channels == self.channels
#         return self.channels
#
#     def _get_values_and_attention(self, ensemble):
#         b, _, n = ensemble.shape
#         d = self.embedding_channels // self.num_heads
#         keys, values = torch.split(self.linear(ensemble), [self.channels, self.embedding_channels], dim=1)
#         keys = torch.reshape(keys, [b, self.num_heads, d, n])
#         values = torch.reshape(values, [b, self.num_heads, self.channels // self.num_heads, n])
#         attention = torch.softmax(
#             torch.sum(self.query[None, ..., None] * keys, dim=2, keepdim=True) / self._sqrt_d,
#             dim=-1
#         )
#         return values, attention
#
#     def forward(self, ensemble):
#         values, attention = self._get_values_and_attention(ensemble)
#         mu = torch.sum(attention * values, dim=-1, keepdim=False).flatten(start_dim=1)
#         if self.keepdim:
#             mu = mu.unsqueeze(-1)
#         return mu


class WeightedMeanStdMerger(WeightedMeanMerger):

    def __init__(
            self, channels: int, embedding_channels: Optional[int] = None, num_heads: int = 1, keepdim=False,
            eps=0., log_std=False, dim=-2
    ):
        super(WeightedMeanMerger, self).__init__(keepdim=keepdim,dim=dim)
        if embedding_channels is None:
            embedding_channels = channels
        assert channels % num_heads == 0, '[ERROR] Input dimension must be divisible by number of heads.'
        assert embedding_channels % num_heads == 0, '[ERROR] Embedding dimension must be divisible by number of heads.'
        self.channels = channels
        self.embedding_channels = embedding_channels
        self.num_heads = num_heads
        self._d = embedding_channels // num_heads
        self._sqrt_d = math.sqrt(self._d)
        self.linear = nn.Conv1d(channels, channels + embedding_channels, (1,))
        self.register_parameter('query', nn.Parameter((torch.rand(num_heads, embedding_channels // num_heads) - 0.5) / 10.))
        self.eps = eps
        self.log_std = log_std

    def output_channels(self, in_channels: int) -> int:
        return 2 * super(WeightedMeanStdMerger, self).output_channels(in_channels)

    def forward(self, ensemble):
        if self.dim == -2:
            ensemble = torch.transpose(ensemble, -1, -2)
        values, attention = self._get_values_and_attention(ensemble)
        mu = torch.sum(attention * values, dim=-1, keepdim=True)
        sigma = torch.sqrt(torch.sum(attention * (values - mu) ** 2, dim=-1, keepdim=True))
        if self.eps > 0:
            sigma = sigma + self.eps
        if self.log_std:
            sigma = torch.log(sigma)
        out = torch.cat([mu, sigma], dim=-3).flatten(start_dim=1)
        if self.keepdim:
            out = out.unsqueeze(self.dim)
        return out


def _test():

    a = torch.randn(10, 20, 4)

    merger = MeanMerger()

    out = merger(a)

    print(out.shape)


if __name__ == '__main__':
    _test()
