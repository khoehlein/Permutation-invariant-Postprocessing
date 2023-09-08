import torch
from torch import nn, Tensor

from numpy import pi as PI


class IYearDayEmbedding(nn.Module):

    def out_channels(self) -> int:
        raise NotImplementedError()


class CyclicYearDayEmbedding(IYearDayEmbedding):

    def __init__(self):
        super(CyclicYearDayEmbedding, self).__init__()

    def forward(self, yday: Tensor):
        yday2pi = 2. * PI * yday
        return torch.stack([torch.sin(yday2pi), torch.cos(yday2pi)], dim=-1)

    def out_channels(self) -> int:
        return 2


class CosineYearDayEmbedding(IYearDayEmbedding):

    def __init__(self):
        super(CosineYearDayEmbedding, self).__init__()

    def forward(self, yday: Tensor):
        yday2pi = 2. * PI * yday
        return torch.cos(yday2pi).unsqueeze(-1)

    def out_channels(self) -> int:
        return 1


class LinearYearDayEmbedding(IYearDayEmbedding):

    def __init__(self):
        super(LinearYearDayEmbedding, self).__init__()

    def forward(self, yday: Tensor):
        return yday[:, None]

    def out_channels(self) -> int:
        return 1


class LearnedYearDayEmbedding(IYearDayEmbedding):

    def __init__(self, num_channels: int = 10, num_nodes: int = 12):
        super(LearnedYearDayEmbedding, self).__init__()
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.register_parameter('embedding', nn.Parameter((torch.rand(num_nodes, num_channels) - 0.5) / 10, requires_grad=True))

    def forward(self, yday: Tensor):
        scaled = (yday * self.num_nodes).view(-1)
        lower = torch.floor(scaled).to(dtype=torch.long)
        upper = (lower + 1) % self.num_nodes
        alpha = (scaled - lower)[:, None]
        emb_lower = self.embedding[lower]
        emb_upper = self.embedding[upper]
        out = alpha * emb_lower + (1 - alpha) * emb_upper
        return out

    def out_channels(self) -> int:
        return self.num_channels
