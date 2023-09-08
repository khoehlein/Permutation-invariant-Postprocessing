import torch
from torch import nn, Tensor
from torch.nn import LayerNorm

import model.custom_activations as ca
from model.ensemble.gaussian_dropout import GaussianDropout


class EnsembleAttentionBlock(nn.Module):

    def __init__(self, channels: int, num_heads: int = 8, dropout=0.):
        super(EnsembleAttentionBlock, self).__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0, \
            f'[ERROR] Model channels ({channels}) must be divisible by number of heads ({num_heads}).'
        self.channels = channels
        self.activation = nn.MultiheadAttention(channels, num_heads, batch_first=True, dropout=dropout)
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


class EnsembleAttention(nn.Module):

    def __init__(
            self,
            num_channels: int = 64,
            num_layers: int = 5, activation='ReLU',
            dropout: float = 0.0, num_heads: int = 8, ff_multiplier=2
    ):
        super(EnsembleAttention, self).__init__()
        assert num_layers % 3 == 0, '[ERROR] Number of layers l must satisfy l = 3 * k for k = 1, 2, 3 ...'
        self.in_channels = num_channels
        self.out_channels = num_channels

        layers = []
        for i in range(num_layers // 3):
            layers += [
                EnsembleAttentionBlock(num_channels, num_heads=num_heads, dropout=dropout),
                FeedForwardBlock(num_channels, multiplier=ff_multiplier, activation=activation),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.model(x)


def _test():
    model = EnsembleAttention(num_channels=64, num_layers=5)
    input = torch.randn(4, 20, 64)
    out = model(input)
    print(out.shape)


if __name__ == '__main__':
    _test()
