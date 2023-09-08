import torch
from torch import nn, Tensor


class BuiltinEmbedding(nn.Module):

    def __init__(self, num_stations, num_features):
        super(BuiltinEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_stations, num_features)

    @property
    def num_stations(self):
        return self.embedding.num_embeddings

    @property
    def num_features(self):
        return self.embedding.embedding_dim

    def forward(self, location: Tensor) -> Tensor:
        return self.embedding(location)

    def out_channels(self):
        return self.embedding.embedding_dim


def _test():
    embedding = BuiltinEmbedding(10, 4)
    x = torch.arange(10)
    out = embedding(x)
    print(out.shape)
    x = x.unsqueeze(-1)
    out = embedding(x)
    print(out.shape)


if __name__ == '__main__':
    _test()