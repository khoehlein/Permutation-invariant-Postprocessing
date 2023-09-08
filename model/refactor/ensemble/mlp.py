from typing import Optional, Dict, Any

import torch
from torch import nn, Tensor

import model.custom_activations as ca
from model.ensemble.gaussian_dropout import GaussianDropout


class IMultiLayerPerceptron(nn.Module):

    def __init__(
            self,
            num_channels: int = 64,
            num_layers: int = 2, dropout=0.0,
            activation: str = 'Softplus', activation_kws: Optional[Dict[str, Any]] = None,
    ):
        super(IMultiLayerPerceptron, self).__init__()
        self.in_channels = num_channels
        self.out_channels = num_channels

        if activation_kws is None:
            activation_kws = {}
        act_class = None
        try:
            act_class = getattr(nn, activation)
        except AttributeError:
            pass
        if act_class is None:
            try:
                act_class = getattr(ca, activation)
            except AttributeError:
                pass
        assert act_class is not None

        def _get_activation():
            return act_class(**activation_kws)

        layers = []
        for i in range(num_layers):
            if dropout > 0.:
                layers.append(self._get_dropout(dropout))
            layers.append(_get_activation())
            layers.append(self._get_linear_layer(num_channels, num_channels))
        layers.append(_get_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _get_linear_layer(self, in_channels, out_channels):
        raise NotImplementedError()

    def _get_dropout(self, dropout):
        raise NotImplementedError()


class MLP(IMultiLayerPerceptron):

    def _get_linear_layer(self, in_channels, out_channels):
        return nn.Linear(in_channels, out_channels)

    def _get_dropout(self, dropout):
        return GaussianDropout(dropout)


class SchulzLerchMLP(nn.Module):

    def __init__(self, mode: str):
        super().__init__()
        if mode == 'bqn':
            self.in_channels = 48
            self.out_channels = 24
        else:
            self.in_channels = 64
            self.out_channels = 32
        self.model = nn.Sequential(
            nn.Softplus(),
            nn.Linear(self.in_channels, self.out_channels),
            nn.Softplus(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def _test():
    model = MLP(num_channels=64, num_layers=4, dropout=0.1)
    print(model)
    input = torch.randn(4, 20, 64)
    out = model(input)
    print(out.shape)


if __name__ == '__main__':
    _test()
