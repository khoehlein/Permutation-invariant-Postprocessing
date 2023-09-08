from math import sqrt
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import LayerNorm

import model.custom_activations as ca
from model.ensemble.gaussian_dropout import GaussianDropout


class ChannelNorm(nn.Module):

    def __init__(self, channels: int, dim: int = 1, eps: float = 1.e-5, affine=True, unbiased=True):
        super(ChannelNorm, self).__init__()
        self.channels = channels
        self.dim = dim
        self.eps = eps
        self.affine = affine
        self.unbiased = unbiased
        self.register_parameter(
            'log_scale',
            nn.Parameter((torch.rand(channels) - 0.5) / 10, requires_grad=True) if affine else None
        )
        self.register_parameter(
            'bias',
            nn.Parameter((torch.rand(channels) - 0.5) / 10, requires_grad=True) if affine else None
        )

    def forward(self, x: Tensor):
        mu = torch.mean(x, dim=self.dim, keepdim=True)
        sigma = torch.std(x, dim=self.dim, unbiased=self.unbiased, keepdim=True)
        x = (x - mu) / (sigma + self.eps)
        if self.affine:
            ex_shape = self._expansion_shape(x)
            x = torch.exp(self.log_scale).view(ex_shape) * x + self.bias.view(ex_shape)
        return x

    def _expansion_shape(self, x:Tensor):
        shape = [1 for _ in x.shape]
        shape[self.dim] = self.channels
        return tuple(shape)

    def extra_repr(self) -> str:
        return '{}, dim={}, eps={}, affine={}, unbiased={}'.format(self.channels, self.dim, self.eps, self.affine, self.unbiased)


class EnsembleAttentionBlock(nn.Module):

    def __init__(self, channels: int, num_heads: int = 8):
        super(EnsembleAttentionBlock, self).__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0, \
            f'[ERROR] Model channels ({channels}) must be divisible by number of heads ({num_heads}).'
        self.channels = channels
        self.activation = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor):
        out = self.activation(x, x, x, need_weights=False)[0]
        return self.norm(x + out)


class FeedForwardBlock(nn.Module):

    def __init__(self, channels, multiplier: int = 4, activation='ReLU', dropout=0.):
        super(FeedForwardBlock, self).__init__()
        layers = [nn.Linear(channels, channels * multiplier)]
        if dropout > 0.:
            layers.append(GaussianDropout(dropout))
        layers += [
            self._get_activation(activation),
            nn.Linear(channels * multiplier, channels),
        ]
        self.model = nn.Sequential(*layers)
        self.norm = LayerNorm(channels) #nn.InstanceNorm1d(channels, affine=True)

    def _get_activation(self, activation):
        act_class = None
        try:
            act_class = getattr(nn, activation)
        except AttributeError:
            act_class = None
        if act_class is None:
            try:
                act_class = getattr(ca, activation)
            except AttributeError:
                raise Exception(f'[ERROR] Activation {activation} neither available in torch.nn nor in customo activations!')
        assert act_class is not None
        return act_class()

    def forward(self, x: Tensor):
        return self.norm(x + self.model(x))


class _EnsembleAttentionBlock(nn.Module):

    def __init__(self, channels: int, hidden_channels: Optional[int] = None, num_heads: int = 8, use_bias=False):
        super(_EnsembleAttentionBlock, self).__init__()
        self.num_heads = num_heads
        if hidden_channels is None:
            hidden_channels = channels
        assert hidden_channels % num_heads == 0, \
            f'[ERROR] Hidden channels ({hidden_channels}) must be divisible by number of heads ({num_heads}).'
        assert channels % num_heads == 0, \
            f'[ERROR] Model channels ({channels}) must be divisible by number of heads ({num_heads}).'
        self.channels = channels
        self.hidden_channels = hidden_channels
        self._sqrt_hidden = sqrt(hidden_channels)
        self.qkv = nn.Conv1d(channels, 2 * hidden_channels + channels, (1,), bias=use_bias)
        self.projection = nn.Conv1d(channels, channels, (1,), bias=use_bias)
        self.norm = ChannelNorm(channels)

    def forward(self, x: Tensor):
        shape = [x.shape[0], self.num_heads, (2 * self.hidden_channels + self.channels) // self.num_heads, x.shape[-1]]
        qkv = self.qkv(x).view(shape)
        splits = [self.hidden_channels // self.num_heads, self.hidden_channels // self.num_heads, self.channels // self.num_heads]
        q, k, v = torch.split(qkv, splits, dim=2)
        q = q.unsqueeze(-2)
        k = k.unsqueeze(-1)
        attention = torch.softmax(torch.sum(k * q, dim=2, keepdim=False) / self._sqrt_hidden, dim=2)
        out = torch.matmul(v, attention)
        out = out.view([x.shape[0], self.channels, -1])
        out = self.projection(out)
        # if self.norm is None:
        #     self.norm = nn.LayerNorm(x.shape[-2:], elementwise_affine=True, device=x.device)
        return self.norm(x + out)


class _FeedForwardBlock(nn.Module):

    def __init__(self, channels, multiplier: int = 1, activation='ReLU', dropout=0.):
        super(_FeedForwardBlock, self).__init__()
        layers = [nn.Conv1d(channels, channels * multiplier, (1,))]
        if dropout > 0.:
            layers.append(GaussianDropout(dropout))
        layers += [
            self._get_activation(activation),
            nn.Conv1d(channels * multiplier, channels, (1,)),
        ]
        self.model = nn.Sequential(*layers)
        self.norm = ChannelNorm(channels) #nn.InstanceNorm1d(channels, affine=True)

    def _get_activation(self, activation):
        act_class = None
        try:
            act_class = getattr(nn, activation)
        except AttributeError:
            act_class = None
        if act_class is None:
            try:
                act_class = getattr(ca, activation)
            except AttributeError:
                raise Exception(f'[ERROR] Activation {activation} neither available in torch.nn nor in customo activations!')
        assert act_class is not None
        return act_class()

    def forward(self, x: Tensor):
        # if self.norm is None:
        #     self.norm = nn.LayerNorm(x.shape[-2:], elementwise_affine=True, device=x.device)
        return self.norm(x + self.model(x))
        # return x + self.model(x)


class EnsembleAttention(nn.Module):

    def __init__(
            self,
            in_channels: int, out_channels: int, hidden_channels: int,
            num_layers: int = 5, activation='ReLU',
            dropout: float = 0.1, num_heads: int = 8, ff_multiplier=4
    ):
        super(EnsembleAttention, self).__init__()
        assert (num_layers - 2) % 3 == 0, '[ERROR] Number of layers l must satisfy l = 3 * k + 2 for k = 1, 2, 3 ...'
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = [nn.Linear(in_channels, hidden_channels)]
        for i in range((num_layers - 2) // 3):
            layers += [
                EnsembleAttentionBlock(hidden_channels, num_heads=num_heads),
                FeedForwardBlock(hidden_channels, multiplier=ff_multiplier, activation=activation, dropout=dropout),
            ]
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.model(x)


def _test():
    model = EnsembleAttention(8, 12, 64, num_layers=5)
    input = torch.randn(4, 20, 8)
    out = model(input)
    print(out.shape)


if __name__ == '__main__':
    _test()
