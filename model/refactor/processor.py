import math
from typing import Optional, Union

from torch import nn, Tensor

from model.ensemble.merger import EnsembleMerger, MeanMerger
from model.refactor.ensemble.attention import EnsembleAttention
from model.refactor.ensemble.mlp import MLP
from model.refactor.ensemble.resnet import ResNet


class Regressor(nn.Module):

    def __init__(self, feature_channels: int, condition_channels: int, output_channels: int, backbone: Union[MLP, ResNet, EnsembleAttention, None]):
        super().__init__()
        self.feature_channels = feature_channels
        self.condition_channels = condition_channels
        self.out_channels = self._compute_output_channels(output_channels, backbone)
        if backbone is None:
            self.backbone = nn.Identity()
            self.output_projection = nn.Identity()
            projection_channels = self.out_channels
        else:
            self.backbone = backbone
            self.output_projection = nn.Linear(backbone.out_channels, self.out_channels)
            projection_channels = backbone.in_channels if backbone is not None else output_channels
        self.feature_projection = nn.Linear(feature_channels, projection_channels)
        self.condition_projection = nn.Linear(condition_channels, projection_channels) if condition_channels > 0 else None

    def _compute_output_channels(self, out_channels, backbone):
        if out_channels is not None:
            return out_channels
        if backbone is not None:
            return backbone.out_channels
        return self.feature_channels + self.condition_channels

    def project(self, features: Tensor, conditions: Tensor):
        projection = self.feature_projection(features)
        if self.condition_projection is not None:
            if len(conditions.shape) != len(features.shape):
                conditions = conditions.unsqueeze(-2)
            projection = projection + self.condition_projection(conditions)
        return projection

    def forward(self, features: Tensor, conditions: Tensor) -> Tensor:
        return self.output_projection(self.backbone(self.project(features, conditions)))


class Rescaler(nn.Module):

    def __init__(self, feature_channels: int, condition_channels: int, output_channels: int):
        assert output_channels is None or output_channels == feature_channels, \
            f'[ERROR] Output channels {output_channels} must equal feature channels {feature_channels}!'
        assert condition_channels == 0, \
            f'[ERROR] Condition channels {condition_channels} must equal zero!'
        super().__init__()
        self.feature_channels = feature_channels
        self.condition_channels = condition_channels
        self.out_channels = feature_channels

    def forward(self, features: Tensor, conditions: Tensor) -> Tensor:
        num_members = features.shape[-2]
        return features * math.sqrt(num_members)


class Identity(nn.Module):

    def __init__(self, feature_channels, condition_channels, output_channels):
        super().__init__()
        assert output_channels is None or (output_channels == feature_channels), \
            f'[ERROR] Output channels {output_channels} must equal feature channels {feature_channels}!'
        assert condition_channels == 0, \
            f'[ERROR] Condition channels {condition_channels} must equal zero!'
        self.feature_channels = feature_channels
        self.condition_channels = condition_channels
        self.out_channels = feature_channels

    def forward(self, features: Tensor, conditions: Tensor) -> Tensor:
        return features


class EnsembleProcessor(nn.Module):

    def __init__(
            self,
            merger: Optional[EnsembleMerger] = None,
            encoder: Optional[Regressor] = None,
            decoder: Optional[Regressor] = None,
    ):
        super(EnsembleProcessor, self).__init__()
        if merger is None:
            merger = MeanMerger(keepdim=False)
        assert not merger.keepdim
        self.merger = merger
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, ensemble: Tensor, conditions: Optional[Tensor] = None, conditions_decoder: Optional[Tensor] = None) -> Tensor:
        if conditions_decoder is None:
            conditions_decoder = conditions
        ensemble_code = self.compute_ensemble_codes(ensemble, conditions=conditions)
        return self.compute_predictions(ensemble_code, conditions=conditions_decoder)

    def compute_predictions(self, ensemble_code: Tensor, conditions = None):
        if self.decoder is not None:
            output = self.decoder(ensemble_code, conditions)
        else:
            output = ensemble_code
        return output

    def compute_member_codes(self, ensemble: Tensor, conditions: Optional[Tensor] = None) -> Tensor:
        if self.encoder is not None:
            member_code = self.encoder(ensemble, conditions)
        else:
            member_code = ensemble
        return member_code

    def compute_ensemble_codes(self, ensemble: Tensor, conditions: Optional[Tensor] = None) -> Tensor:
        member_code = self.compute_member_codes(ensemble, conditions)
        ensemble_code = self.merger(member_code)
        return ensemble_code
