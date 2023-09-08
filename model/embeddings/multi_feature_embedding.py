from typing import Dict

import torch
from torch import nn


class MultiFeatureEmbedding(nn.Module):

    def __init__(self, module_mapping: Dict[str, nn.Module]):
        super().__init__()
        self.module_mapping = nn.ModuleDict({k: m for k, m in module_mapping.items() if m is not None})
        self.module_keys = list(sorted(self.module_mapping.keys()))
        self._num_out_channels = sum([
            module_mapping[key].out_channels()
            for key in self.module_mapping.keys()
        ])

    def out_channels(self):
        return self._num_out_channels

    def forward(self, **kwargs):
        out = None
        current_channel = 0
        for key in self.module_keys:
            key_data = kwargs[key]
            if out is None:
                out = torch.empty(len(key_data), self._num_out_channels, device=key_data.device, dtype=key_data.dtype)
            module = self.module_mapping[key]
            key_channels = module.out_channels()
            features = module(key_data)
            out[..., current_channel:(current_channel + key_channels)] = features
            current_channel = current_channel + key_channels
        return out
