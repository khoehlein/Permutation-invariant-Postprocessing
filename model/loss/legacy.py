import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributions as dist

from model.loss.bernstein.integration import UniformQuantileIntegrator, MidpointQuantileIntegrator, \
    GaussLegendreQuantileIntegrator
from model.loss.losses import LogisticDistribution, TruncatedLogisticDistribution, BernsteinCRPS


class LogisticCRPS(nn.Module):

    def __init__(self, eps=1.e-6, reduction='mean', positive_constraint='softplus', max_negative_mu_scale=0.):
        super(LogisticCRPS, self).__init__()
        self.eps = eps
        self.dist = LogisticDistribution()
        self.reduction = {'mean': torch.mean, 'sum': torch.sum, 'none': nn.Identity()}[reduction]
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]
        self.max_negative_mu_scale = float(max_negative_mu_scale)

    def in_channels(self):
        return 2

    def compute_parameters(self, predictions: Tensor, merge=False):
        mu, sigma = torch.unbind(self.positive_constraint(predictions), dim=1)
        if self.eps > 0.:
            sigma = sigma + self.eps
        if hasattr(self, 'max_negative_mu_scale'):
            if self.max_negative_mu_scale > 0.:
                mu = mu - self.max_negative_mu_scale * sigma
        out = (mu, sigma)
        if merge:
            out = torch.stack(out, dim=1)
        return out

    def forward(self, predictions, observations):
        mu, sigma = self.compute_parameters(predictions)
        if len(observations.shape) > 1:
            observations = observations[:, 0]
        if len(mu.shape) != len(observations.shape):
            assert len(mu.shape) == len(observations.shape) + 1
            mu = torch.mean(mu, dim=-1)
            sigma = torch.mean(sigma, dim=-1)
        res = self.compute_crps(mu, sigma, observations)
        return self.reduction(res)

    def compute_crps(self, mu, sigma, observations):
        z = torch.clamp(observations, min=0.)
        z_0 = - mu / sigma
        z_y = (z - mu) / sigma
        lp_0 = self.dist.log_cdf(z_0)
        p_0 = torch.exp(lp_0)
        lp_m0 = self.dist.log_cdf(-z_0)
        p_m0 = torch.exp(lp_m0)
        lp_my = self.dist.log_cdf(-z_y)
        b = lp_m0 - (1. + 2. * lp_my) / p_m0 - torch.square(p_0) * lp_0 / torch.square(p_m0)
        crps = torch.abs(z - observations) - (z - mu) * (1. + p_0) / p_m0 + sigma * b
        return crps

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1, unsqueeze=True):
        mu, sigma = self.compute_parameters(predictions)
        mu = mu[..., None]
        sigma = sigma[..., None]
        z_0 = - mu / sigma
        F_0 = self.dist.cdf(z_0)
        p = torch.rand(*F_0.shape[:-1], num_samples, device=F_0.device, dtype=F_0.dtype)
        z = - torch.log1p(- F_0) - torch.log1p(- p) + torch.log((1. - F_0) * p + F_0)
        x = sigma * z + mu
        if unsqueeze:
            x = x.unsqueeze(1)
        return x


class LogisticLogScore(nn.Module):

    def __init__(self, eps=1.e-6, reduction='mean', positive_constraint='softplus', max_negative_mu_scale=0.):
        super(LogisticLogScore, self).__init__()
        self.eps = eps
        self.dist = LogisticDistribution()
        self.reduction = {'mean': torch.mean, 'sum': torch.sum, 'none': nn.Identity()}[reduction]
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]
        self.max_negative_mu_scale = max_negative_mu_scale

    def in_channels(self):
        return 2

    def compute_parameters(self, predictions: Tensor, merge=False):
        mu, sigma = torch.unbind(self.positive_constraint(predictions), dim=1)
        if self.eps > 0.:
            sigma = sigma + self.eps
        if hasattr(self, 'max_negative_mu_scale'):
            if self.max_negative_mu_scale > 0.:
                mu = mu - self.max_negative_mu_scale * sigma
        out = (mu, sigma)
        if merge:
            out = torch.stack(out, dim=1)
        return out

    def forward(self, predictions, observations):
        mu, sigma = self.compute_parameters(predictions)
        if len(observations.shape) > 1:
            observations = observations[:, 0]
        if len(mu.shape) != len(observations.shape):
            assert len(mu.shape) == len(observations.shape) + 1
            mu = torch.mean(mu, dim=-1)
            sigma = torch.mean(sigma, dim=-1)
        z = torch.clamp(observations, min=0.)
        mu_red = mu / sigma
        z_y = (z - mu) / sigma
        log_p = self.dist.log_pdf(z_y) - torch.log(sigma)
        log_trunc_norm = - F.softplus(- mu_red) # mu_red - softplus(mu_red) = - softplus(-mu_red)
        res = - (log_p - log_trunc_norm)
        return self.reduction(res)

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1, unsqueeze=True):
        mu, sigma = self.compute_parameters(predictions)
        mu = mu[..., None]
        sigma = sigma[..., None]
        z_0 = - mu / sigma
        F_0 = self.dist.cdf(z_0)
        p = torch.rand(*F_0.shape[:-1], num_samples, device=F_0.device, dtype=F_0.dtype)
        z = - torch.log1p(- F_0) - torch.log1p(- p) + torch.log((1. - F_0) * p + F_0)
        x = sigma * z + mu
        if unsqueeze:
            x = x.unsqueeze(1)
        return x


class NormalCRPS(nn.Module):

    def __init__(self, reduction='mean', positive_constraint='exp', eps=1.e-6):
        super().__init__()
        self.eps = eps
        self.dist = dist.Normal(0., 1.)
        self._inv_sqrt_pi = 1. / math.sqrt(np.pi)
        self.reduction = {'mean': torch.mean, 'sum': torch.sum, 'none': nn.Identity()}[reduction]
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]

    def in_channels(self):
        return 2

    def compute_parameters(self, predictions: Tensor, merge=False):
        mu, log_sigma = torch.unbind(predictions, dim=1)
        sigma = self.positive_constraint(log_sigma)
        if self.eps > 0.:
            sigma = sigma + self.eps
        out = (mu, sigma)
        if merge:
            out = torch.stack(out, dim=1)
        return out

    def forward(self, predictions, observations):
        mu, sigma = self.compute_parameters(predictions)
        if len(observations.shape) > 1:
            observations = observations[:, 0]
        if len(mu.shape) != len(observations.shape):
            assert len(mu.shape) == len(observations.shape) + 1
            mu = torch.mean(mu, dim=-1)
            sigma = torch.mean(sigma, dim=-1)
        res = self.compute_crps(mu, sigma, observations)
        return self.reduction(res)

    def compute_crps(self, mu, sigma, observations):
        z_red = (observations - mu) / sigma
        cdf = self.dist.cdf(z_red)
        pdf = torch.exp(self.dist.log_prob(z_red))
        crps = sigma * (z_red * (2. * cdf - 1.) + 2. * pdf - self._inv_sqrt_pi)
        return crps

    def sample_posterior(self, predictions: Tensor, num_samples: int = 1, unsqueeze=True):
        mu, sigma = self.compute_parameters(predictions)
        mu = mu[..., None]
        sigma = sigma[..., None]
        z = torch.randn(sigma.shape[:-1], num_samples, device=sigma.device, dtype=torch.float32)
        x = sigma * z + mu
        if unsqueeze:
            x = x.unsqueeze(1)
        return x


class LogNormalCRPS(nn.Module):

    def __init__(self, parameterization='direct', reduction='mean', positive_constraint='softplus'):
        # Formula from https://arxiv.org/pdf/1407.3252.pdf
        super(LogNormalCRPS, self).__init__()
        self.dist = torch.distributions.Normal(0., 1.)
        self.parameterize = {'direct': self._direct_params, 'transformed': self._transformed_params}[parameterization]
        self.reduction = {'mean': torch.mean, 'sum': torch.sum}[reduction]
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]
        self._inv_sqrt_2 = 1. / math.sqrt(2.)

    def in_channels(self):
        return 2

    def forward(self, predictions, observations):
        mu, std, m = self.parameterize(predictions)
        log_obs_reduced = (torch.log(observations) - mu) / std
        f = self.dist.cdf(log_obs_reduced)
        crps = observations * (2. * f - 1.) - 2. * m * (
                    self.dist.cdf(log_obs_reduced - std) + self.dist.cdf(std * self._inv_sqrt_2) - 1.)
        return self.reduction(crps)

    def _direct_params(self, predictions):
        mu, log_std = torch.unbind(predictions, dim=-1)
        std = self.positive_constraint(log_std)
        m = torch.exp(mu + std ** 2 / 2.)
        return mu, std, m

    def _transformed_params(self, predictions):
        log_m, log_v = torch.chunk(predictions, 2, dim=-1)
        mu = 2. * log_m - 0.5 * (log_v + F.softplus(2. * log_m - log_v))
        std = torch.sqrt(F.softplus(log_v - 2. * log_m))
        m = torch.exp(log_m)
        return mu, std, m


class BernsteinQuantileCRPS(nn.Module):

    def __init__(self, degree: int = 12, num_quantiles: int = 99, reduction='mean', positive_constraint='softplus', integration_scheme='uniform'):
        super(BernsteinQuantileCRPS, self).__init__()
        self.degree = degree
        self.num_quantiles = num_quantiles
        self.integrator = self._get_integrator(integration_scheme)
        self.register_buffer('polynomials',self._compute_polynomials())
        self.reduction = {'mean': torch.mean, 'sum': torch.sum, 'none': nn.Identity()}[reduction]
        self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]

    def _get_integrator(self, scheme: str):
        if scheme == 'uniform':
            return UniformQuantileIntegrator(self.num_quantiles)
        elif scheme == 'midpoint' or scheme == 'rectangle':
            return MidpointQuantileIntegrator(self.num_quantiles)
        elif scheme == 'gauss-legendre' or scheme == 'legendre-gauss':
            return GaussLegendreQuantileIntegrator(self.num_quantiles)
        else:
            raise NotImplementedError(f'[ERROR] Encountered unknown integration scheme: {scheme}.')

    def _compute_polynomials(self, quantiles = None):
        if quantiles is None:
            quantiles = self.integrator.nodes
        quantiles = quantiles[None, :]
        device = quantiles.device
        k = torch.arange(self.degree + 1, device=device)[:, None]
        n = torch.full_like(k, self.degree)
        log_quantiles = torch.log(quantiles)
        log_quantiles_inv = torch.log1p(-quantiles)
        log_binom = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        log_polynomials = log_binom + k * log_quantiles + (n - k) * log_quantiles_inv
        return torch.exp(log_polynomials)

    def in_channels(self):
        return self.degree + 1

    def compute_parameters(self, predictions: Tensor, merge=False):
        weights = self.positive_constraint(predictions)
        weights = torch.cumsum(weights, dim=1)
        return weights

    def compute_quantiles(self, *params, p=None):
        weights = params[0]
        if p is None:
            polynomials = self.polynomials
        else:
            polynomials = self._compute_polynomials(quantiles=p)
        if len(weights.shape) == 2:
            q = torch.matmul(weights, polynomials)
        else:
            q = torch.einsum('bi...,ij->bj...', weights, polynomials)
        return q

    def forward(self, predictions: Tensor, observations: Tensor) -> Tensor:
        weights = self.compute_parameters(predictions)
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(1)
        if len(weights.shape) != len(observations.shape):
            assert len(weights.shape) == len(observations.shape) + 1
            weights = torch.mean(weights, dim=-1)
        q = self.compute_quantiles(weights)
        crps = self.compute_crps(q, observations)
        return self.reduction(crps)

    def compute_crps(self, q, observations):
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(1)
        p = self.integrator.nodes[None, :]
        err = observations - q
        e1 = err * p
        e2 = err * (p - 1.)
        crps = self.integrator(2. * torch.maximum(e1, e2))
        return crps


class BernsteinTransformedLogisticQuantileCRPS(BernsteinQuantileCRPS):

    def __init__(self, degree: int = 12, num_quantiles: int = 99, reduction='mean', positive_constraint='softplus', integration_scheme='uniform', eps=1.e-6, max_negative_mu_scale=0.):
        super(BernsteinTransformedLogisticQuantileCRPS, self).__init__(degree, num_quantiles, reduction, positive_constraint, integration_scheme)
        self.eps = eps
        self.max_negative_mu_scale = float(max_negative_mu_scale)

    def in_channels(self):
        return self.degree + 2

    def _compute_bernstein_weights(self, raw_weights: Tensor):
        shape = list(raw_weights.shape)
        shape[1] = self.degree + 1
        weights = torch.zeros(shape, device=raw_weights.device, dtype=raw_weights.dtype)
        weights[:, 1:] = torch.cumsum(self.positive_constraint(raw_weights), dim=1)
        weights = weights / weights[:, -1].unsqueeze(1)
        return weights

    def _compute_logistic_parameters(self, predictions: Tensor):
        mu, sigma = torch.unbind(self.positive_constraint(predictions), dim=1)
        if self.eps > 0.:
            sigma = sigma + self.eps
        if hasattr(self, 'max_negative_mu_scale'):
            if self.max_negative_mu_scale > 0.:
                mu = mu - self.max_negative_mu_scale * sigma
        return mu, sigma

    def compute_parameters(self, predictions: Tensor, merge=False):
        logistic_params, bernstein_weights = torch.split(predictions, [2, self.degree], dim=1)
        bernstein_weights = self._compute_bernstein_weights(bernstein_weights)
        mu, sigma = self._compute_logistic_parameters(logistic_params)
        out = (mu.unsqueeze(1), sigma.unsqueeze(1), bernstein_weights)
        if merge:
            out = torch.cat(out, dim=1)
        return out

    def forward(self, predictions: Tensor, observations: Tensor) -> Tensor:
        mu, sigma, bernstein_weights = self.compute_parameters(predictions)
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(1)
        if len(bernstein_weights.shape) != len(observations.shape):
            assert len(bernstein_weights.shape) == len(observations.shape) + 1
            bernstein_weights = torch.mean(bernstein_weights, dim=-1)
            mu = torch.mean(mu, dim=-1)
            sigma = torch.mean(sigma, dim=-1)
        q = self.compute_quantiles(mu, sigma, bernstein_weights)
        crps = self.compute_crps(q, observations)
        return self.reduction(crps)

    def compute_quantiles(self, *params, p=None):
        mu, sigma, bernstein_weights = params
        dist = TruncatedLogisticDistribution(location=mu, scale=sigma)
        q = dist.quantile(super().compute_quantiles(bernstein_weights, p=p))
        return q


class BernsteinTransformedExponentialQuantileCRPS(BernsteinQuantileCRPS):

    def __init__(self, degree: int = 12, num_quantiles: int = 99, reduction='mean', positive_constraint='softplus', integration_scheme='uniform', eps=1.e-6, max_negative_mu_scale=0.):
        super(BernsteinTransformedExponentialQuantileCRPS, self).__init__(degree, num_quantiles, reduction, positive_constraint, integration_scheme)
        self.eps = eps
        self.max_negative_mu_scale = float(max_negative_mu_scale)

    def in_channels(self):
        return self.degree + 1

    def _compute_bernstein_weights(self, raw_weights: Tensor):
        shape = list(raw_weights.shape)
        shape[1] = self.degree + 1
        weights = torch.zeros(shape, device=raw_weights.device, dtype=raw_weights.dtype)
        weights[:, 1:] = torch.cumsum(self.positive_constraint(raw_weights), dim=1)
        weights = weights / weights[:, -1].unsqueeze(1)
        return weights

    def _compute_exponential_parameters(self, predictions: Tensor):
        sigma = torch.unbind(self.positive_constraint(predictions), dim=1)[0]
        if self.eps > 0.:
            sigma = sigma + self.eps
        return sigma

    def compute_parameters(self, predictions: Tensor, merge=False):
        exp_params, bernstein_weights = torch.split(predictions, [1, self.degree], dim=1)
        bernstein_weights = self._compute_bernstein_weights(bernstein_weights)
        sigma = self._compute_exponential_parameters(exp_params)
        out = (sigma.unsqueeze(1), bernstein_weights)
        if merge:
            out = torch.cat(out, dim=1)
        return out

    def forward(self, predictions: Tensor, observations: Tensor) -> Tensor:
        sigma, bernstein_weights = self.compute_parameters(predictions)
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(1)
        if len(bernstein_weights.shape) != len(observations.shape):
            assert len(bernstein_weights.shape) == len(observations.shape) + 1
            bernstein_weights = torch.mean(bernstein_weights, dim=-1)
            sigma = torch.mean(sigma, dim=-1)
        q = self.compute_quantiles(sigma, bernstein_weights)
        crps = self.compute_crps(q, observations)
        return self.reduction(crps)

    def compute_quantiles(self, *params, p=None):
        sigma, bernstein_weights = params
        p_bernstein = super().compute_quantiles(bernstein_weights, p=p)
        q = - torch.log1p(-p_bernstein) / sigma
        return q


def _test():
    loss = BernsteinQuantileCRPS(reduction='none')
    inputs = torch.randn(10, loss.in_channels(),requires_grad=True)
    loss_new = BernsteinCRPS(reduction='none')

    print(loss.in_channels(), loss_new.in_channels())

    targets = torch.randn(10, 1) ** 2
    uni = loss(inputs, targets)
    uni_new = loss_new(inputs, targets)
    print(uni.shape)
    # loss = BernsteinQuantileCRPS(reduction='none', integration_scheme='midpoint')
    # mpr = loss(inputs, targets)
    # loss = BernsteinQuantileCRPS(reduction='none', integration_scheme='gauss-legendre')
    # glq = loss(inputs, targets)
    #
    # q = (mpr + glq) / 2.
    #
    # def relative_deviation(a, b):
    #     return 2 * torch.abs(a - b) / torch.sqrt(a * b)
    #
    # print(relative_deviation(glq, mpr))
    # print(relative_deviation(uni, q))



if __name__ == '__main__':
    _test()

# class PiecewiseLinearCRPS(nn.Module):
#
#     def __init__(self, positive_constraint='softplus'):
#         super(PiecewiseLinearCRPS, self).__init__()
#         self.positive_constraint = {'softplus': F.softplus, 'exp': torch.exp}[positive_constraint]
#
#     def forward(self, predictions, observations):
#         if len(predictions.shape) == len(observations.shape):
#             assert observations.shape[-1] == 1,\
#                 '[ERROR] Piecewise linear CRPS not suitable for multivariate observations!'
#             observations = observations[..., 0]
#         num_samples, num_channels = predictions.shape
#         sample_idx = torch.arange(num_samples)
#         assert num_channels % 2 == 1, '[ERROR] Number of prediction channels must be odd!'
#         assert num_channels > 5
#         num_bins = num_channels // 2
#         log_diffs, log_pdf_raw = torch.split(predictions, [num_bins, num_bins + 1], dim=-1)
#         nodes = torch.zeros_like(log_pdf_raw)
#         diffs = self.positive_constraint(log_diffs)
#         nodes[..., 1:] = torch.cumsum(diffs, dim=-1)
#         pdf, cdf = self._get_distributions(log_pdf_raw)
#         position = torch.relu(torch.searchsorted(nodes, observations) - 1)
#         observed_cdf = cdf[sample_idx, position]
#         observation_in_range = observations < nodes[:, -1]
#         if torch.any(observation_in_range):
#             residual_cdf = self._compute_residual_cdf(
#                 pdf[observation_in_range, ...],
#                 position[observation_in_range],
#                 nodes[observation_in_range],
#                 observations[observation_in_range]
#             )
#             observed_cdf[observation_in_range] = observed_cdf[observation_in_range] + residual_cdf
#
#     def _compute_residual_cdf(self, pdf, position, nodes):
#
#
#
#     def _get_distributions(self, log_pdf_raw):
#         pdf_raw = self.positive_constraint(log_pdf_raw)
#         cdf = torch.zeros_like(pdf_raw)
#         cum_pdf = torch.cumsum((pdf_raw[..., :-1] + pdf_raw[..., 1:]) / 2., dim=-1)
#         norm = cum_pdf[..., -1][..., None]
#         pdf = pdf_raw / norm
#         cdf[..., 1:] = cum_pdf / norm
#         return pdf, cdf
# def phi(x):
#     return (1. + torch.special.erf(x / sqrt(2.))) / 2.
#
#
# def compute_crps(predictions: Tensor, observations: Tensor, norm=None) -> object:
#     log_m, log_v = torch.chunk(predictions, 2, dim=-1)
#     # mu = 2. * log_m - 0.5 * (log_v + F.softplus(2. * log_m - log_v))
#     # std = torch.sqrt(F.softplus(log_v - 2. * log_m))
#     # m = torch.exp(log_m)
#     mu, log_std = torch.chunk(predictions, 2, dim=-1)
#     std = torch.exp(log_std)
#     m = torch.exp(mu + std ** 2 / 2.)
#     log_obs_reduced = (torch.log(observations) - mu) / std
#     crps = observations * (2. * phi(log_obs_reduced) - 1.) - 2. * m * (phi(log_obs_reduced - std) + phi(std / sqrt(2.)) - 1.)
#     loss = torch.sum(crps)
#     return loss

