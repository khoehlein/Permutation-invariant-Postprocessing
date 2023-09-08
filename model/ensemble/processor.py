from typing import Optional

import torch
from torch import nn, Tensor

from model.ensemble.merger import EnsembleMerger, MeanMerger
from model.ensemble.mlp import ScalarMLP, EnsembleMLP


class EnsembleProcessor(nn.Module):

    def __init__(
            self,
            merger: Optional[EnsembleMerger] = None,
            encoder: Optional[EnsembleMLP] = None,
            decoder: Optional[ScalarMLP] = None,
            conditions_at_input: bool = False,
            conditions_at_bottleneck: bool = False,
    ):
        super(EnsembleProcessor, self).__init__()
        self.conditions_at_input = conditions_at_input
        self.conditions_at_bottleneck = conditions_at_bottleneck
        if merger is None:
            merger = MeanMerger(keepdim=False)
        assert not merger.keepdim
        self.merger = merger
        if encoder is None:
            assert not conditions_at_input
        self.encoder = encoder
        if decoder is None:
            assert not conditions_at_bottleneck
        self.decoder = decoder

    def forward(self, ensemble: Tensor, conditions: Optional[Tensor] = None) -> Tensor:
        ensemble_code = self.compute_ensemble_codes(ensemble, conditions=conditions)
        return self.compute_predictions(ensemble_code, conditions=conditions)

    def compute_predictions(self, ensemble_code: Tensor, conditions = None, conditions_decoder = None):
        if self.decoder is not None:
            if self.conditions_at_bottleneck:
                assert conditions is not None
                predictors = self._combine_inputs(ensemble_code, conditions)
            else:
                predictors = ensemble_code
            output = self.decoder(predictors)
        else:
            output = ensemble_code
        return output

    def compute_member_codes(self, ensemble: Tensor, conditions: Optional[Tensor] = None) -> Tensor:
        if self.conditions_at_input:
            assert conditions is not None
            predictors = self._combine_inputs(ensemble, conditions)
        else:
            predictors = ensemble
        member_code = self.encoder(predictors)
        return member_code

    def compute_ensemble_codes(self, ensemble: Tensor, conditions: Optional[Tensor] = None) -> Tensor:
        if self.encoder is not None:
            member_code = self.compute_member_codes(ensemble, conditions)
        else:
            member_code = ensemble
        ensemble_code = self.merger(member_code)
        return ensemble_code

    @staticmethod
    def _combine_inputs(x: Tensor, conditions: Tensor):
        if len(x.shape) != len(conditions.shape):
            conditions = conditions.unsqueeze(-2)
            tile_shape = [1 for _ in x.shape]
            tile_shape[-2] = x.shape[-2]
            conditions = conditions.repeat(*tile_shape)
        predictors = torch.cat([x, conditions], dim=-1)
        return predictors