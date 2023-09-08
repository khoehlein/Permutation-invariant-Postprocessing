import math

import numpy as np
import pandas as pd

import torch
from scipy.stats import logistic, norm
from torch import nn, Tensor
from torch.distributions import Normal
from torch.nn import functional as F

from experiments.baselines.common import Coverage


def np_softplus(x: np.ndarray):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.)


class LogisticDistribution(nn.Module):

    def __init__(self, loc=0., scale=1., _softmax_threshold=20.):
        super().__init__()
        self.location = float(loc)
        self.scale = float(scale)
        self._log_scale = math.log(self.scale)
        self._softmax_threshold = _softmax_threshold

    def cdf(self, x: Tensor) -> Tensor:
        # return 0.5 * (1. + torch.tanh((x - self.location) / (2. * self.scale)))
        return torch.exp(self.log_cdf(x))

    def log_cdf(self, x: Tensor) -> Tensor:
        z = (x - self.location) / self.scale
        return - F.softplus(-z, beta=1., threshold=self._softmax_threshold)

    def log_pdf(self, x: Tensor) -> Tensor:
        z = (x - self.location) / self.scale
        log_pdf = - z - self._log_scale - 2. * F.softplus(- z, beta=1., threshold=self._softmax_threshold)
        return log_pdf

    def quantile(self, p: Tensor) -> Tensor:
        z = - torch.log1p(-p) + torch.log(p)
        x = self.scale * z + self.location
        return x


def crps_logistic(y_pred, y_true):
    tfd = logistic()
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]

    z = np.maximum(0., y_true.ravel())
    z_0 = - mu / sigma
    z_y = (z - mu) / sigma

    p_0 = tfd.cdf(z_0)
    lp_0 = tfd.logcdf(z_0)
    p_m0 = tfd.cdf(-z_0)
    lp_m0 = tfd.logcdf(-z_0)
    lp_my = tfd.logcdf(-z_y)

    b = lp_m0 - (1. + 2. * lp_my) / p_m0 - (p_0 ** 2.) * lp_0 / (p_m0 ** 2.)

    res = np.abs(z - y_true.ravel()) - (z - mu) * (1. + p_0) / p_m0 + sigma * b
    return pd.Series(data=res, name='CRPS')


def nll_logistic(y_pred, y_true):
    tfd = logistic()
    mu, sigma = y_pred[:, 0], y_pred[:, 1]
    if len(y_true.shape) != len(mu.shape):
        y_true = y_true.unsqueeze(-1)
    z = np.maximum(y_true, 0.)
    mu_red = mu / sigma
    z_y = (z - mu) / sigma
    log_p = tfd.logpdf(z_y) - np.log(sigma)
    log_trunc_norm = mu_red - np_softplus(mu_red)  # mu_red - softplus(mu_red) = - softplus(-mu_red)
    score = - (log_p - log_trunc_norm)
    return pd.Series(data=score, name='NLL')


def pit_logistic(y_pred, y_true):
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    dist = logistic
    F_y = dist.cdf(y_true, loc=mu, scale=sigma)
    y_0 = np.zeros_like(y_true)
    SF_0 = dist.sf(y_0, loc=mu, scale=sigma)
    F_0 = dist.cdf(y_0, loc=mu, scale=sigma)
    return (F_y - F_0) / SF_0


tfd = LogisticDistribution(loc=0., scale=1.)

def crps_logistic_torch(y_pred, y_true):
    mu, sigma = torch.chunk(y_pred, 2, dim=-1)
    if len(y_true.shape) != len(mu.shape):
        y_true = y_true.unsqueeze(-1)
    z = torch.maximum(torch.zeros_like(y_true), y_true)
    z_0 = - mu / sigma
    z_y = (z - mu) / sigma
    p_0 = tfd.cdf(z_0)
    lp_0 = tfd.log_cdf(z_0)
    p_m0 = tfd.cdf(-z_0)
    lp_m0 = tfd.log_cdf(-z_0)
    lp_my = tfd.log_cdf(-z_y)
    b = lp_m0 - (1. + 2. * lp_my) / p_m0 - torch.square(p_0) * lp_0 / torch.square(p_m0)
    res = torch.abs(z - y_true) - (z - mu) * (1. + p_0) / p_m0 + sigma * b
    res = torch.mean(res)
    return res


def nll_logistic_torch(y_pred, y_true):
    mu, sigma = torch.chunk(y_pred, 2, dim=-1)
    if len(y_true.shape) != len(mu.shape):
        y_true = y_true.unsqueeze(-1)
    z = torch.clamp(y_true, min=0.)
    mu_red = mu / sigma
    z_y = (z - mu) / sigma
    log_p = tfd.log_pdf(z_y) - torch.log(sigma)
    log_trunc_norm = mu_red - F.softplus(mu_red)  # mu_red - softplus(mu_red) = - softplus(-mu_red)
    score = - (log_p - log_trunc_norm)
    return torch.mean(score)


tfd_normal = Normal(0., 1.)

def crps_normal_torch(y_pred, y_true):
    mu, sigma = torch.chunk(y_pred, 2, dim=-1)
    if len(y_true.shape) != len(mu.shape):
        y_true = y_true.unsqueeze(-1)
    sigma = sigma + 1.e-4
    z_red = (y_true - mu) / sigma
    cdf = tfd_normal.cdf(z_red)
    pdf = torch.exp(tfd_normal.log_prob(z_red))
    crps = sigma * (z_red * (2. * cdf - 1.) + 2. * pdf - 1. / math.sqrt(np.pi))
    return torch.mean(crps)


def nll_normal_torch(y_pred, y_true):
    mu, sigma = torch.chunk(y_pred, 2, dim=-1)
    if len(y_true.shape) != len(mu.shape):
        y_true = y_true.unsqueeze(-1)
    z_red = (y_true - mu) / sigma
    log_pdf = tfd_normal.log_prob(z_red) - torch.log(sigma)
    return - torch.mean(log_pdf)


def crps_normal(y_pred, y_true):
    tfd = norm()
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    z_red = (y_true.ravel() - mu) / sigma
    cdf = tfd.cdf(z_red)
    pdf = tfd.pdf(z_red)
    crps = sigma * (z_red * (2. * cdf - 1.) + 2. * pdf - 1. / math.sqrt(np.pi))
    return pd.Series(data=crps, name='CRPS')


def nll_normal(y_pred, y_true):
    tfd = norm()
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    z_red = (y_true.ravel() - mu) / sigma
    log_pdf = tfd.logpdf(z_red) - np.log(sigma)
    return pd.Series(data=-log_pdf, name='NLL')


def pit_normal(y_pred: np.ndarray, y_true: np.ndarray):
    dist = norm()
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    y_red = (y_true - mu) / sigma
    F_y = dist.cdf(y_red)
    y_0 = -mu / sigma
    SF_0 = dist.sf(y_0)
    F_0 = dist.cdf(y_0)
    return (F_y - F_0) / SF_0


class _LocScalePrediction(object):

    def __init__(self, data: np.ndarray, loc_positive=True):
        assert len(data.shape) == 2
        assert data.shape[-1] == 2
        assert np.all(np.logical_or(data[:, -1] >= 0, np.isclose(data[:, -1], 0)))
        data[:, -1] = np.maximum(data[:, -1], 1.e-6)
        if loc_positive:
            assert np.all(np.logical_or(data[:, 0] >= 0, np.isclose(data[:, 0], 0)))
            data[:, 0] = np.maximum(data[:, 0], 1.e-6)
        self.data = data

    def _quantile(self, p: float, tfd):
        mu = self.data[:, 0]
        sigma = self.data[:, 1]
        z_0 = np.zeros_like(mu)
        F_0 = tfd.cdf(z_0, loc=mu, scale=sigma)
        SF_0 = tfd.sf(z_0, loc=mu, scale=sigma)
        p_raw = p * SF_0 + F_0
        return tfd.ppf(p_raw, loc=mu, scale=sigma)

    def compute_crps(self, observations: np.ndarray):
        raise NotImplementedError()

    def compute_pit(self, observations: np.ndarray):
        raise NotImplementedError()

    def compute_pi_length(self, alpha: float = 0.05):
        raise NotImplementedError()

    def num_samples(self):
        return len(self.data)

    def compute_coverage(self, observations: np.ndarray, alpha: float = 0.05):
        pit = self.compute_pit(observations)
        return Coverage(alpha).from_pit(pit)


class LogisticPrediction(_LocScalePrediction):

    def compute_crps(self, observations: np.ndarray):
        return crps_logistic(self.data, observations)

    def compute_log_score(self, observations: np.ndarray):
        return nll_logistic(self.data, observations)

    def compute_pit(self, observations: np.ndarray):
        return pit_logistic(self.data, observations)

    def compute_pi_length(self, alpha: float = 0.05):
        p_lower, p_upper = alpha /2., 1. - alpha / 2.
        return self._quantile(p_upper, logistic) - self._quantile(p_lower, logistic)


class NormalPrediction(_LocScalePrediction):

    def compute_crps(self, observations: np.ndarray):
        return crps_normal(self.data, observations)

    def compute_pit(self, observations: np.ndarray):
        return pit_normal(self.data, observations)

    def compute_pi_length(self, alpha: float = 0.05):
        p_lower, p_upper = alpha / 2., 1. - alpha / 2.
        return self._quantile(p_upper, norm) - self._quantile(p_lower, norm)