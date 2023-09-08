from typing import Optional, Dict, Any

from torch import nn, Tensor

from model.ensemble.gaussian_dropout import GaussianDropout
import model.custom_activations as ca


class IResNet(nn.Module):

    class ResidualBlock(nn.Module):

        def __init__(self, channels: int, layer_func, activation_func, dropout_func, dropout=0.):
            super(IResNet.ResidualBlock, self).__init__()
            layers = []
            layers.append(layer_func(channels, channels))
            layers.append(activation_func())
            if dropout > 0:
                layers.append(dropout_func(dropout))
            layers.append(layer_func(channels, channels))
            layers.append(activation_func())
            self.model = nn.Sequential(*layers)

        def forward(self, x: Tensor):
            return x + self.model(x)

    def __init__(
            self,
            in_channels: int, out_channels: int, hidden_channels: int,
            num_layers: int = 3, dropout=0.1,
            activation: str = 'LeakyRelu', activation_kws: Optional[Dict[str, Any]] = None,
    ):
        super(IResNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert num_layers % 2 == 0

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

        layers = [
            self._get_linear_layer(in_channels, hidden_channels),
            _get_activation(),
        ]
        if dropout > 0:
            layers.append(self._get_dropout(dropout))
        for i in range((num_layers - 2) // 2):
            layers.append(
                self.ResidualBlock(
                    hidden_channels,
                    self._get_linear_layer, _get_activation, self._get_dropout,
                    dropout=dropout
                )
            )
        layers.append(self._get_linear_layer(hidden_channels, out_channels))
        model = nn.Sequential(*layers)
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _get_linear_layer(self, in_channels, out_channels):
        raise NotImplementedError()

    def _get_dropout(self, dropout):
        raise NotImplementedError()


class ScalarResNet(IResNet):

    def _get_linear_layer(self, in_channels, out_channels):
        return nn.Linear(in_channels, out_channels)

    def _get_dropout(self, dropout):
        return GaussianDropout(dropout)


class EnsembleResNet(IResNet):

    def _get_linear_layer(self, in_channels, out_channels):
        return nn.Linear(in_channels, out_channels)

    def _get_dropout(self, dropout):
        return GaussianDropout(dropout)
