import torch
from torch import nn


class UniformDequantization(nn.Module):

    def forward(self, x):
        noise = (torch.rand_like(x) - 0.5) / 2.
        return torch.clip(x + noise, min=0)
