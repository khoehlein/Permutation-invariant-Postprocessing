import torch
from torch import nn


class StationEmbedding(nn.Module):

    def __init__(self, num_stations, num_features):
        super(StationEmbedding, self).__init__()
        self.register_parameter('embedding', nn.Parameter((torch.rand(num_stations, num_features) - 0.5) / 10., requires_grad=True))

    @property
    def num_stations(self):
        return self.embedding.shape[0]

    @property
    def num_features(self):
        return self.embedding.shape[1]

    def forward(self, location):
        return self.embedding[location, :]

    def out_channels(self):
        return self.embedding.shape[-1]

    def extra_repr(self) -> str:
        return 'num_stations={}, num_features={}'.format(self.num_stations, self.num_features)


def _test():
    embedding = StationEmbedding(10, 4)
    x = torch.arange(10)
    out = embedding(x)
    print(out.shape)
    x = x.unsqueeze(-1)
    out = embedding(x)
    print(out.shape)


if __name__ == '__main__':
    _test()
