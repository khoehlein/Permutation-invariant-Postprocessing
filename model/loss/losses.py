import math
from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributions as dist

from model.loss.bernstein.integration import (
    UniformQuantileIntegrator,
    MidpointQuantileIntegrator,
    GaussLegendreQuantileIntegrator,
)


class LogisticDistribution(nn.Module):

    def __init__(self, location=0., scale=1., _softmax_threshold=20.):
        super().__init__()
        self.register_buffer('location', torch.as_tensor(location))
        self.register_buffer('scale', torch.as_tensor(scale))
        self.register_buffer('_log_scale', torch.log(self.scale))
        self._softmax_threshold = _softmax_threshold

    def cdf(self, x: Tensor):
        # return 0.5 * (1. + torch.tanh((x - self.location) / (2. * self.scale)))
        return torch.exp(self.log_cdf(x))

    def log_sf(self, x: Tensor):
        x_red = (x - self.location) / self.scale
        return - F.softplus(x_red, beta=1., threshold=self._softmax_threshold)

    def sf(self, x: Tensor):
        return torch.exp(self.log_sf(x))

    def log_cdf(self, x: Tensor):
        x_red = (x - self.location) / self.scale
        return - F.softplus(- x_red, beta=1., threshold=self._softmax_threshold)

    def log_pdf(self, x: Tensor):
        x_red = (x - self.location) / self.scale
        log_pdf = - x_red - self._log_scale - 2. * F.softplus(- x_red, beta=1., threshold=self._softmax_threshold)
        return log_pdf

    def quantile(self, p: Tensor):
        z = torch.logit(p)
        x = self.scale * z + self.location
        return x

# x - log(1 + exp(x))

class TruncatedLogisticDistribution(object):

    def __init__(self, location=0., scale=1., _softmax_threshold=20.):
        self.dist = LogisticDistribution(location, scale, _softmax_threshold)

    @property
    def location(self):
        return self.dist.location

    @property
    def scale(self):
        return self.dist.scale

    def log_cdf(self, x: Tensor):
        log_F_0 = self.dist.log_cdf(torch.zeros_like(x))
        log_F_x = self.dist.log_cdf(x)
        return log_F_x + torch.log1p(-torch.exp(log_F_0 - log_F_x)) - torch.log1p(-torch.exp(log_F_0))

    def cdf(self, x: Tensor):
        return torch.exp(self.log_cdf(x))

    def log_pdf(self, x: Tensor):
        log_F_0 = self.dist.log_cdf(torch.zeros_like(x))
        log_pdf = self.dist.log_pdf(x) - torch.log1p(-torch.exp(log_F_0))
        return log_pdf

    def quantile(self, p: Tensor):
        F_0 = self.dist.cdf(torch.zeros_like(p))
        p_red = (1. - F_0) * p + F_0
        x = self.dist.quantile(p_red)
        return x


class IScoringRule(nn.Module):

    def __init__(self, in_channels: int, reduction='mean'):
        super().__init__()
        self._in_channels = in_channels
        self.reduction = {'mean': torch.mean, 'sum': torch.sum, 'none': nn.Identity()}[reduction]

    def in_channels(self) -> int:
        return self._in_channels

    def compute_parameters(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        # should return parameters in format predictions.shape[:-1] + channel, with singleton channels unsqueezed
        raise NotImplementedError()

    def compute_score(self, observations: Tensor, *params: Tensor) -> Tensor:
        # should receive parameters from above and work with observations with unsqueezed squeezed unit dim
        raise NotImplementedError()

    def forward(self, predictions: Tensor, observations: Tensor):
        params = self.compute_parameters(predictions, merge=False)
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(-1)
        assert observations.shape[-1] == 1
        score = self.compute_score(observations, *params)
        return self.reduction(score)

    def sample_posterior(self, predictions: Tensor, num_samples: int=1 ) -> Tensor:
        raise NotImplementedError()

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        raise NotImplementedError()


class LogisticParameters(nn.Module):

    def __init__(self, eps=0., positive_constraint='softplus', max_negative_mu_scale=0.):
        super().__init__()
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]
        self.max_negative_mu_scale = float(max_negative_mu_scale)
        self.eps = eps
        self.dist = LogisticDistribution()

    def in_channels(self):
        return 2

    def forward(self, predictions: Tensor, merge=False) -> Tuple[Tensor, Tensor]:
        mu, sigma = torch.chunk(self.positive_constraint(predictions), self.in_channels(), dim=1)
        if self.eps > 0.:
            sigma = sigma + self.eps
        if hasattr(self, 'max_negative_mu_scale'):
            if self.max_negative_mu_scale > 0.:
                mu = mu - self.max_negative_mu_scale * sigma
        out = (mu, sigma)
        if merge:
            out = torch.cat(out, dim=-1)
        return out

    def sample_posterior(self, predictions: Tensor, num_samples=1):
        mu, sigma = self.forward(predictions)
        mu = mu.unsqueeze(-2)
        sigma = sigma.unsqueeze(-2)
        p = torch.rand(*mu.shape[:-1], num_samples, device=mu.device, dtype=mu.dtype)
        return self.compute_quantiles(p, mu, sigma)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        mu, sigma = params
        z_0 = - mu / sigma
        F_0 = self.dist.cdf(z_0)
        SF_0 = self.dist.sf(z_0)
        p_red = SF_0 * p + F_0
        x = sigma * torch.logit(p_red) + mu
        return x


class LogisticLogParameters(nn.Module):

    def __init__(self):
        super().__init__()
        self.dist = LogisticDistribution()

    def in_channels(self):
        return 2

    def forward(self, predictions: Tensor, merge=False) -> Tuple[Tensor, Tensor]:
        log_mu, log_sigma = torch.chunk(predictions, self.in_channels(), dim=1)
        out = (log_mu, log_sigma)
        if merge:
            out = torch.cat(out, dim=-1)
        return out

    def sample_posterior(self, predictions: Tensor, num_samples=1):
        log_mu, log_sigma = self.forward(predictions)
        log_mu = log_mu.unsqueeze(-2)
        log_sigma = log_sigma.unsqueeze(-2)
        p = torch.rand(*log_mu.shape[:-1], num_samples, device=log_mu.device, dtype=log_mu.dtype)
        return self.compute_quantiles(p, log_mu, log_sigma)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        log_mu, log_sigma = params
        z_0 = - torch.exp(log_mu - log_sigma)
        F_0 = self.dist.cdf(z_0)
        SF_0 = self.dist.sf(z_0)
        p_red = SF_0 * p + F_0
        z_red = torch.log(p_red) - torch.log1p(- p_red)
        x = torch.exp(log_sigma) * z_red + torch.exp(log_mu)
        return x


class LogisticCRPS(IScoringRule):

    def __init__(self, eps = 0., reduction='mean', positive_constraint='softplus', max_negative_mu_scale=0.):
        params = LogisticParameters(eps, positive_constraint, max_negative_mu_scale)
        super().__init__(params.in_channels(), reduction)
        self.parameterization = params

    def compute_parameters(self, predictions: Tensor, merge=False):
        return self.parameterization(predictions, merge)

    def compute_score(self, observations: Tensor, *params: Tensor):
        mu, sigma = params
        z = torch.clamp(observations, min=0.)
        z_0 = - mu / sigma
        z_y = (z - mu) / sigma
        lp_0 = self.parameterization.dist.log_cdf(z_0)
        p_0 = torch.exp(lp_0)
        lp_m0 = self.parameterization.dist.log_cdf(-z_0)
        p_m0 = torch.exp(lp_m0)
        lp_my = self.parameterization.dist.log_cdf(-z_y)
        b = lp_m0 - (1. + 2. * lp_my) / p_m0 - torch.square(p_0) * lp_0 / torch.square(p_m0)
        crps = torch.abs(z - observations) - (z - mu) * (1. + p_0) / p_m0 + sigma * b
        return crps

    def sample_posterior(self, predictions: Tensor, num_samples: int=1) -> Tensor:
        return self.parameterization.sample_posterior(predictions, num_samples)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        return self.parameterization.compute_quantiles(p, *params)


class LogisticLogScore(IScoringRule):

    def __init__(self, eps=0., reduction='mean', positive_constraint='softplus', max_negative_mu_scale=0.):
        params = LogisticLogParameters()
        super().__init__(params.in_channels(), reduction)
        self.parameterization = params

    def compute_parameters(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        return self.parameterization(predictions, merge)

    def compute_score(self, observations: Tensor, *params: Tensor) -> Tensor:
        log_mu, log_sigma = params
        z = torch.clamp(observations, min=0.)
        mu = torch.exp(log_mu)
        sigma = torch.exp(log_sigma)
        mu_red = torch.exp(log_mu - log_sigma)
        z_y = (z - mu) / sigma
        log_p = self.parameterization.dist.log_pdf(z_y) - log_sigma
        log_trunc_norm = mu_red - F.softplus(mu_red)  # mu_red - softplus(mu_red) = - softplus(-mu_red)
        score = - (log_p - log_trunc_norm)
        return score

    def sample_posterior(self, predictions: Tensor, num_samples: int=1) -> Tensor:
        return self.parameterization.sample_posterior(predictions, num_samples)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        return self.parameterization.compute_quantiles(p, *params)


class NormalParameters(nn.Module):

    def __init__(self, positive_constraint='exp', eps=1.e-6):
        super().__init__()
        self.eps = eps
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]
        self.dist = dist.Normal(0., 1.)

    def in_channels(self):
        return 2

    def forward(self, predictions: Tensor, merge=False):
        mu, raw_sigma = torch.chunk(predictions, self.in_channels(), dim=1)
        sigma = self.positive_constraint(raw_sigma)
        if self.eps > 0.:
            sigma = sigma + self.eps
        out = (mu, sigma)
        if merge:
            out = torch.cat(out, dim=1)
        return out

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1):
        mu, sigma = self(predictions)
        mu = mu[..., None]
        sigma = sigma[..., None]
        z = torch.randn(sigma.shape[:-1], num_samples, device=sigma.device, dtype=torch.float32)
        x = sigma * z + mu
        return x

    def compute_quantiles(self, p:Tensor, *params: Tensor) -> Tensor:
        mu, sigma = params
        return mu + sigma * math.sqrt(2.) * torch.erfinv(2. * p - 1.)


class NormalCRPS(IScoringRule):

    def __init__(self, reduction='mean', positive_constraint='softplus', eps=1.e-6):
        params = NormalParameters(positive_constraint, eps)
        super().__init__(params.in_channels(), reduction)
        self.parameterization = params
        self._inv_sqrt_pi = 1. / math.sqrt(np.pi)

    def compute_score(self, observations: Tensor, *params: Tensor) -> Tensor:
        mu, sigma = params
        z_red = (observations - mu) / sigma
        cdf = self.parameterization.dist.cdf(z_red)
        pdf = torch.exp(self.parameterization.dist.log_prob(z_red))
        crps = sigma * (z_red * (2. * cdf - 1.) + 2. * pdf - self._inv_sqrt_pi)
        return crps

    def compute_parameters(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        return self.parameterization(predictions, merge)

    def sample_posterior(self, predictions: Tensor, num_samples: int=1 ) -> Tensor:
        return self.parameterization.sample_posterior(predictions, num_samples)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        return self.parameterization.compute_quantiles(p, *params)


class NormalLogScore(IScoringRule):

    def __init__(self, reduction='mean', positive_constraint='exp', eps=1.e-6):
        params = NormalParameters(positive_constraint, eps)
        super().__init__(params.in_channels(), reduction)
        self.dist = dist.Normal(0., 1.)
        self.parameterization = params

    def compute_score(self, observations: Tensor, *params: Tensor) -> Tensor:
        mu, sigma = params
        z_red = (observations - mu) / sigma
        score = self.dist.log_prob(z_red) - torch.log(sigma)
        return score

    def compute_parameters(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        return self.parameterization(predictions, merge)

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1) -> Tensor:
        return self.parameterization.sample_posterior(predictions, num_samples)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        return self.parameterization.compute_quantiles(p, *params)


class LogNormalParameters(NormalParameters):

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1):
        samples = super().sample_posterior(predictions, num_samples)
        return torch.exp(samples)

    def compute_quantiles(self, p:Tensor, *params: Tensor) -> Tensor:
        samples = super().compute_quantiles(p, *params)
        return torch.exp(samples)


class LogNormalCRPS(IScoringRule):

    def __init__(self, reduction='mean', positive_constraint='exp', eps=1.e-6):
        params = LogNormalParameters(positive_constraint, eps)
        super().__init__(params.in_channels(), reduction)
        self.parameterization = params
        self._inv_sqrt_2 = 1. / math.sqrt(2.)

    def compute_score(self, observations: Tensor, *params: Tensor) -> Tensor:
        mu, std = params
        m = torch.exp(mu + std ** 2 / 2.)
        log_obs_reduced = (torch.log(observations) - mu) / std
        f = self.parameterization.dist.cdf(log_obs_reduced)
        crps = observations * (2. * f - 1.) - 2. * m * (
                    self.parameterization.dist.cdf(log_obs_reduced - std) +
                    self.parameterization.dist.cdf(std * self._inv_sqrt_2) - 1.
        )
        return crps

    def compute_parameters(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        return self.parameterization(predictions, merge)

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1) -> Tensor:
        return self.parameterization.sample_posterior(predictions, num_samples)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        return self.parameterization.compute_quantiles(p, *params)


class LogNormalLogScore(IScoringRule):

    def __init__(self, reduction='mean', positive_constraint='exp', eps=1.e-6):
        params = LogNormalParameters(positive_constraint, eps)
        super().__init__(params.in_channels(), reduction)
        self.parameterization = params
        self._inv_sqrt_2 = 1. / math.sqrt(2.)

    def compute_score(self, observations: Tensor, *params: Tensor) -> Tensor:
        mu, sigma = params
        log_sigma = torch.log(sigma)
        log_obs = torch.log(observations)
        log_obs_red = (log_obs -mu) / sigma
        score = self.parameterization.dist.log_prob(log_obs_red) - log_sigma - log_obs
        return score

    def compute_parameters(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        return self.parameterization(predictions, merge)

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1) -> Tensor:
        return self.parameterization.sample_posterior(predictions, num_samples)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        return self.parameterization.compute_quantiles(p, *params)


class BernsteinWeights(nn.Module):

    def __init__(self, degree, normalize=True, pad_zero=False, positive_constraint='softplus'):
        super().__init__()
        self.degree = degree
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]
        self.pad_zero = pad_zero
        self.normalize = normalize

    def in_channels(self):
        return self.degree + int(not self.pad_zero)

    def forward(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        shape = list(predictions.shape)
        shape[-1] = self.degree + 1
        weights = torch.zeros(shape, device=predictions.device, dtype=predictions.dtype)
        weights[..., int(self.pad_zero):] = torch.cumsum(self.positive_constraint(predictions), dim=-1)
        if self.normalize:
            weights = weights / weights[..., [-1]]
        if merge:
            return weights
        weights = (weights,)
        return weights


class BernsteinQuantiles(nn.Module):

    @staticmethod
    def compute_polynomials(degree: int, nodes: Tensor):
        nodes = nodes.unsqueeze(-2)
        shape = [1 for _ in  nodes.shape]
        shape[-2] = degree + 1
        device = nodes.device
        k = torch.arange(degree + 1, device=device).view(*shape)
        n = torch.full_like(k, degree)
        log_quantiles = torch.log(nodes)
        log_quantiles_inv = torch.log1p(-nodes)
        log_binom = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        log_polynomials = log_binom + k * log_quantiles + (n - k) * log_quantiles_inv
        return torch.exp(log_polynomials)

    def __init__(self, degree: int, nodes: Tensor):
        super().__init__()
        assert len(nodes.shape) == 1
        self.degree = degree
        self.register_buffer('polynomials', self.compute_polynomials(degree, nodes))

    def forward(self, weights: Tensor, polynomials: Optional[Tensor] = None):
        if polynomials is None:
            polynomials = self.polynomials
        quantiles = torch.matmul(weights.unsqueeze(-2), polynomials)
        return quantiles.squeeze(-2)


def _build_integrator(scheme: str, num_nodes: int):
    if scheme == 'uniform':
        return UniformQuantileIntegrator(num_nodes)
    elif scheme == 'midpoint' or scheme == 'rectangle':
        return MidpointQuantileIntegrator(num_nodes)
    elif scheme == 'gauss-legendre' or scheme == 'legendre-gauss':
        return GaussLegendreQuantileIntegrator(num_nodes)
    else:
        raise NotImplementedError(f'[ERROR] Encountered unknown integration scheme: {scheme}.')


class BernsteinCRPS(IScoringRule):

    def __init__(self, degree: int = 12, num_quantiles: int = 99, reduction='mean', positive_constraint='softplus', integration_scheme='uniform'):
        params = BernsteinWeights(degree, pad_zero=False, normalize=False, positive_constraint=positive_constraint)
        super().__init__(params.in_channels(), reduction)
        self.parameterization = params
        self.integrator = _build_integrator(integration_scheme, num_quantiles)
        self.quantiles = BernsteinQuantiles(degree, self.integrator.nodes)

    def compute_score(self, observations: Tensor, *params: Tensor) -> Tensor:
        weights = params[0]
        q = self.quantiles(weights)
        p = self.integrator.nodes.unsqueeze(0)
        err = observations - q
        e1 = err * p
        e2 = err * (p - 1.)
        crps = self.integrator(2. * torch.maximum(e1, e2))
        return crps

    def compute_parameters(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        return self.parameterization(predictions, merge=merge)

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        weights = params[0]
        polynomials = self.quantiles.compute_polynomials(self.quantiles.degree, p)
        return self.quantiles(weights, polynomials=polynomials)

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1) -> Tensor:
        p = torch.rand(*predictions.shape[:-1], num_samples, device=predictions.device, dtype=predictions.dtype)
        parameters = self.parameterization.forward(predictions)
        return self.compute_quantiles(p, *parameters)


class JointParameters(object):

    def __init__(self, bqn: BernsteinWeights, other):
        self.bqn = bqn
        self.other = other
        self._num_channels = [bqn.in_channels(), other.in_channels()]

    def in_channels(self):
        return sum(self._num_channels)

    def forward(self, predictions: Tensor, merge=False):
        bqn, other = torch.split(predictions, self._num_channels, dim=-1)
        out = (*self.bqn(bqn), *self.other(other))
        if merge:
            out = torch.cat(out, dim=-1)
        return out


class ExponentialParameters(nn.Module):

    def __init__(self, eps=1.e-6, positive_constraint='softplus'):
        super().__init__()
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]
        self.eps = eps

    def in_channels(self):
        return 1

    def forward(self, predictions: Tensor, merge=False) -> Tuple[Tensor]:
        return (self.positive_constraint(predictions) + self.eps,)

    def sample_posterior(self, predictions: Tensor, num_samples=1):
        gamma = self.forward(predictions)[0].unsqueeze(-2)
        p = torch.rand(*gamma.shape[:-1], num_samples, device=gamma.device, dtype=gamma.dtype)
        return self.compute_quantiles(p, gamma)

    def compute_quantiles(self, p: Tensor, *parameters: Tensor) -> Tensor:
        gamma = parameters[0]
        return - torch.log1p(- p) * gamma


class _BernsteinTransformedCRPS(IScoringRule):

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1) -> Tensor:
        p = torch.rand(*predictions.shape[:-1], num_samples, device=predictions.device, dtype=predictions.dtype)
        parameters = self.parameterization.forward(predictions)
        return self.compute_quantiles(p, *parameters)

    def __init__(self, bqn_weights: BernsteinWeights, other_params: Union[NormalParameters, LogNormalParameters, LogisticParameters, ExponentialParameters], reduction='mean', integration_scheme='uniform', num_quantiles=99):
        assert bqn_weights.pad_zero == True
        params = JointParameters(bqn_weights, other_params)
        super().__init__(params.in_channels(), reduction)
        self.parameterization = params
        self.integrator = _build_integrator(integration_scheme, num_quantiles)
        self.quantiles = BernsteinQuantiles(params.bqn.degree, self.integrator.nodes)

    def compute_parameters(self, predictions: Tensor, merge=False) -> Tuple[Tensor, ...]:
        return self.parameterization.forward(predictions, merge)

    def compute_score(self, observations: Tensor, *params: Tensor) -> Tensor:
        weights, *other_params = params
        qp = self.quantiles(weights)
        q = self.parameterization.other.compute_quantiles(qp, *other_params)
        p = self.integrator.nodes.unsqueeze(0)
        err = observations - q
        e1 = err * p
        e2 = err * (p - 1.)
        crps = self.integrator(2. * torch.maximum(e1, e2))
        return crps

    def compute_quantiles(self, p: Tensor, *params: Tensor) -> Tensor:
        weights, *other_params = params
        polynomials = self.quantiles.compute_polynomials(self.quantiles.degree, p)
        qp = self.quantiles(weights, polynomials=polynomials)
        return self.parameterization.other.compute_quantiles(qp, *other_params)


class BernsteinTransformedLogisticCRPS(_BernsteinTransformedCRPS):

    def __init__(
            self,
            eps=1.e-6, max_negative_mu_scale=0.,
            degree: int = 12, positive_constraint='softplus',
            reduction='mean', integration_scheme='uniform', num_quantiles=99,
    ):
        other_params = LogisticParameters(eps, positive_constraint, max_negative_mu_scale)
        bqn_params = BernsteinWeights(degree, normalize=True, pad_zero=True, positive_constraint=positive_constraint)
        super().__init__(bqn_params, other_params, reduction, integration_scheme, num_quantiles)


class BernsteinTransformedExponentialCRPS(_BernsteinTransformedCRPS):

    def __init__(
            self,
            eps=1.e-6,
            degree: int = 12, positive_constraint='softplus',
            reduction='mean', integration_scheme='uniform', num_quantiles=99,
    ):
        other_params = ExponentialParameters(eps, positive_constraint)
        bqn_params = BernsteinWeights(degree, normalize=True, pad_zero=True, positive_constraint=positive_constraint)
        super().__init__(bqn_params, other_params, reduction, integration_scheme, num_quantiles)


def _test():
    batch_size = 128
    num_quantiles = 10
    observations = torch.randn(batch_size, 1) ** 2
    rule = BernsteinTransformedLogisticCRPS()
    predictions = torch.randn(batch_size, rule.in_channels())

    params = rule.compute_parameters(predictions)
    score = rule.compute_score(observations, *params)
    quantiles = rule.compute_quantiles(torch.rand(1, num_quantiles), *params)

    print('Done')


if __name__ == '__main__':
    _test()
