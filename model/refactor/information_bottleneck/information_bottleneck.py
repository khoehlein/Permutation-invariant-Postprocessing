import ast
import math
from typing import Dict, Union

import torch
from torch import nn, Tensor

from model.helpers import get_activation
from model.custom_activations import Sine


class MultiGroupLinear(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_groups: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        amplitude = math.sqrt(6. / in_channels)
        self.register_parameter(
            'weight',
            nn.Parameter(
                2. * amplitude * (torch.rand(num_groups, in_channels, out_channels) - 0.5),
                requires_grad=True
            )
        )
        nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
        self.register_parameter(
            'bias',
            nn.Parameter(
                0.1 * (torch.rand(num_groups, out_channels) - 0.5),
                requires_grad=True
            )
        )

    def forward(self, x: Tensor):
        out = torch.matmul(x.unsqueeze(-2), self.weight).squeeze(-2)
        leading_dims = [1 for _ in out.shape[:-2]]
        return out + self.bias.view(*leading_dims, *self.bias.shape)


class Siren(nn.Module):

    def __init__(self, num_groups: int, num_channels: int = 32, num_layers: int = 1):
        super().__init__()
        # https://arxiv.org/pdf/2006.09661.pdf
        self.in_channels = num_channels
        self.out_channels = num_channels
        layers = [Sine(omega=30.)]
        for _ in range(num_layers):
            layers.append(MultiGroupLinear(num_channels, num_channels, num_groups))
            layers.append(Sine(omega=1.))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.model(x)


class MultiGroupMLP(nn.Module):

    def __init__(self, num_groups: int, num_channels: int = 32, num_layers: int = 1, activation='LeakyReLU'):
        super().__init__()
        self.in_channels = num_channels
        self.out_channels = num_channels
        layers = [get_activation(activation)]
        for _ in range(num_layers):
            layers.append(MultiGroupLinear(num_channels, num_channels, num_groups))
            layers.append(get_activation(activation))
        self.model = nn.Sequential(*layers)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, MultiGroupLinear):
            nn.init.uniform_(m.weight.data, -0.05, 0.05)
            nn.init.uniform_(m.bias.data, -0.05, 0.05)

    def forward(self, x: Tensor):
        return self.model(x)


class FourierEmbedding(nn.Module):

    def __init__(self, num_freqs: int = 4):
        super().__init__()
        self.register_buffer('frequencies', 2. ** torch.arange(num_freqs))

    def out_channels(self):
        return 1 + 2 * len(self.frequencies)

    def forward(self, x: Tensor):
        x = x.unsqueeze(-1)
        x_scaled = x * self.frequencies
        return torch.cat([x, torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)


class InformationBottleneckV1(nn.Module):

    def __init__(self, num_groups: int, embedding_channels: int, backbone: Union[Siren, MultiGroupMLP]):
        super().__init__()
        self.in_channels = num_groups
        self.out_channels = embedding_channels * num_groups
        self.num_groups = num_groups
        self.backbone = backbone
        self.embedd = FourierEmbedding(4)
        self.encode = MultiGroupLinear(self.embedd.out_channels(), backbone.in_channels, num_groups)
        self.decode_mu = MultiGroupLinear(backbone.out_channels, embedding_channels, num_groups)
        self.decode_log_var = MultiGroupLinear(backbone.out_channels, embedding_channels, num_groups)
        # nn.init.normal_(self.decode_log_var.weight.data, 0., 1.e-6)
        # nn.init.normal_(self.decode_log_var.bias.data, 0., 1.e-6)

    def parameterize(self, x: Tensor):
        features = self.encode(self.embedd(x))
        features = self.backbone(features)
        mu = self.decode_mu(features)
        log_var = torch.clamp(self.decode_log_var(features), min=-6.)
        return mu, log_var

    def sample_posterior(self, mu: Tensor, log_var: Tensor):
        sigma = torch.exp(log_var / 2.)
        return mu + sigma * torch.randn_like(sigma)

    def compute_kl_divergence(self, mu: Tensor, log_var: Tensor):
        return 0.5 * torch.sum(torch.exp(log_var) + torch.square(mu) - 1. - log_var, dim=-1)

    def forward(self, x: Tensor):
        mu, log_var = self.parameterize(x)
        d_kl = self.compute_kl_divergence(mu, log_var)
        samples = self.sample_posterior(mu, log_var).flatten(start_dim=-2)
        return samples, d_kl

    @classmethod
    def from_args(cls, args, num_groups):
        kwargs = ast.literal_eval(args['model:bottleneck:backbone_kws'])
        backbone = MultiGroupMLP(num_groups, **kwargs)
        return cls(num_groups, args['model:bottleneck:embedding_dim'], backbone)


class InformationBottleneckV2(nn.Module):

    def __init__(self, num_groups: int, embedding_channels: int, backbone: Union[Siren, MultiGroupMLP]):
        super().__init__()
        self.in_channels = num_groups
        self.out_channels = embedding_channels * num_groups
        self.num_groups = num_groups
        self.backbone = backbone
        self.embedd = FourierEmbedding(4)
        self.encode = MultiGroupLinear(self.embedd.out_channels(), backbone.in_channels, num_groups)
        self.decode_embedding = MultiGroupLinear(backbone.out_channels, embedding_channels, num_groups)
        self.decode_weighting = MultiGroupLinear(backbone.out_channels, 2, num_groups)
        self.norm = nn.LayerNorm(embedding_channels, elementwise_affine=False)

    def parameterize(self, x: Tensor):
        features = self.encode(self.embedd(x))
        features = self.backbone(features)
        embedding = self.norm(self.decode_embedding(features))
        log_weights = torch.log_softmax(torch.clamp(self.decode_weighting(features), -6., 6.), dim=-1)
        return embedding, log_weights

    def sample_posterior(self, embedding: Tensor, log_weights: Tensor):
        weights = torch.exp(log_weights)
        noise = torch.randn_like(embedding)
        eweights = weights[..., [0]]
        nweights = weights[..., [1]]
        return embedding * eweights + noise * nweights

    def compute_kl_divergence(self, embedding: Tensor, log_weights: Tensor):
        eweights = torch.exp(log_weights[..., [0]])
        log_var = 2. * log_weights[..., [1]]
        mu = embedding * eweights
        return 0.5 * torch.sum(torch.exp(log_var) + torch.square(mu) - 1. - log_var, dim=-1)

    def forward(self, x: Tensor):
        mu, log_var = self.parameterize(x)
        d_kl = self.compute_kl_divergence(mu, log_var)
        samples = self.sample_posterior(mu, log_var).flatten(start_dim=-2)
        return samples, d_kl

    @classmethod
    def from_args(cls, args, num_groups):
        kwargs = ast.literal_eval(args['model:bottleneck:backbone_kws'])
        backbone = MultiGroupMLP(num_groups, **kwargs)
        return cls(num_groups, args['model:bottleneck:embedding_dim'], backbone)


class MultiFeatureBottleneck(nn.Module):

    def __init__(self, module_mapping: Dict[str, InformationBottleneckV1]):
        super().__init__()
        self.module_mapping = nn.ModuleDict(module_mapping)
        self.keys = list(self.module_mapping.keys())

    def forward(self, **kwargs):
        all_samples, all_d_kl = [], []
        for key in self.keys:
            samples, d_kl = self.module_mapping[key](kwargs[key])
            all_samples.append(samples)
            all_d_kl.append(d_kl)
        return torch.cat(all_samples, dim=-1), torch.cat(all_d_kl, dim=-1)


def _test():
    siren = MultiGroupMLP(3)
    ib = InformationBottleneckV1(3, 8, siren)
    x = torch.randn(10, 20, 3)
    out, _ = ib(x)
    print(out.shape)


if __name__ == '__main__':
    _test()
