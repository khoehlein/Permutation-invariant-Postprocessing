
import torch
import torch.nn as nn


class StandardScaler(nn.Module):

    def __init__(self, channel_dim=-1, eps=1.e-6):
        super(StandardScaler, self).__init__()
        self._channel_dim = channel_dim
        self.register_buffer('_num_samples', None)
        self.register_buffer('_mu', None)
        self.register_buffer('_m2', None)
        self._eps = eps

    def channel_dim(self, shape):
        shape_dim = len(shape)
        if self._channel_dim >= shape_dim or self._channel_dim < -shape_dim:
            raise Exception(f'[ERROR] Channel dim {self._channel_dim} invalid for data of shape {shape}')
        return self._channel_dim if self._channel_dim > 0 else len(shape) + self._channel_dim

    def _get_summation_dims(self, shape):
        cd = self.channel_dim(shape)
        return [i for i in range(len(shape)) if i != cd]

    def fit(self, data):
        shape = data.shape
        summation_dims = self._get_summation_dims(shape)
        valid = ~ torch.isnan(data)
        num_new_samples = torch.sum(valid, dim=summation_dims, keepdim=True)
        sample_mu = torch.sum(torch.where(valid, data, torch.zeros_like(data)), dim=summation_dims, keepdim=True)
        sample_mu = sample_mu / num_new_samples
        sample_m2 = torch.sum(torch.where(valid, torch.square(data - sample_mu), torch.zeros_like(data)), dim=summation_dims, keepdim=True)
        if self._num_samples is None:
            self._mu = sample_mu
            self._m2 = sample_m2
            self._num_samples = num_new_samples
        else:
            delta = sample_mu - self._mu
            self._mu = self._mu + delta * (num_new_samples / (self._num_samples + num_new_samples))
            self._m2 = self._m2 + sample_m2 + delta ** 2 * ((num_new_samples * self._num_samples) / (num_new_samples + self._num_samples))
            self._num_samples = self._num_samples + num_new_samples
        return self

    def mean(self):
        return self._mu

    def std(self, unbiased=True):
        if self._num_samples is None or torch.min(self._num_samples) < 2:
            raise Exception('[ERROR] Standard scaler requires at least two samples to determine standard deviation')
        norm = self._num_samples - 1 if unbiased else self._num_samples
        return self._m2 / norm

    def transform(self, data):
        return (data - self._mu) / (self.std(unbiased=True) + self._eps)

    def revert(self, data):
        return (self.std(unbiased=True) + self._eps) * data + self._mu

    def reset(self):
        self._num_samples = 0
        self._mu = None
        self._m2 = None
        return self
