import numpy as np
from .StatisticsTracker import StatisticsTracker
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
# Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products".
# Technometrics. 4 (3): 419–420. doi:10.2307/1266577. JSTOR 1266577


class WelfordStatisticsTracker(StatisticsTracker):
    def __init__(self, count=0, weight_sum=0, sum_of_squared_weights=0, mean=0, sum_of_squared_values=0):
        super(WelfordStatisticsTracker, self).__init__(count)
        self._weight_sum = weight_sum
        self._sum_of_squared_weights = sum_of_squared_weights
        self._mean = mean
        self._sum_of_squared_values = sum_of_squared_values

    def get_state(self):
        return self._count, self._weight_sum, self._sum_of_squared_weights, self._mean, self._sum_of_squared_values

    def set_state(self, count=None, weight_sum=None, sum_of_squared_weights=None, mean=None, sum_of_squared_values=None):
        if count is not None:
            self._count = count
        if weight_sum is not None:
            self._weight_sum = weight_sum
        if sum_of_squared_weights is not None:
            self._sum_of_squared_weights = sum_of_squared_weights
        if mean is not None:
            self._mean = mean
        if sum_of_squared_values is not None:
            self._sum_of_squared_values = sum_of_squared_values
        return self

    def reset(self):
        return self.set_state(count=0, weight_sum=0, sum_of_squared_weights=0, mean=0, sum_of_squared_values=0)

    def update(self, value, weight=1.):
        self._count += 1
        self._weight_sum += weight
        self._sum_of_squared_weights += weight**2
        delta = value - self._mean
        self._mean += (weight / self._weight_sum) * delta
        delta2 = value - self._mean
        self._sum_of_squared_values += weight * delta * delta2
        return self

    def summary(self, unbiased=True, weight_mode='frequency'):
        return self.count(), self.mean(), self.var(unbiased=unbiased, weight_mode=weight_mode)

    def count(self):
        return self._count

    def mean(self):
        return self._mean

    def _compute_variance(self, unbiased=True, weight_mode='frequency'):
        assert self._weight_sum > 0, self._count > 1
        correction = 0.
        if unbiased:
            if weight_mode == 'frequency':
                assert self._weight_sum % 1 == 0., '[ERROR] Sum of weights found inconsistent with frequency weighting.'
                correction = 1.
            elif weight_mode == 'reliability':
                correction = self._sum_of_squared_weights / self._weight_sum
            else:
                raise NotImplementedError('[ERROR] Unknown weight-mode specification: {}'.format(weight_mode))
        return self._sum_of_squared_values / (self._weight_sum - correction)

    def var(self, unbiased=True, weight_mode='frequency'):
        # Note that this function does not account for weighting appropriately
        if self._count < 2:
            return np.nan
        return self._compute_variance(unbiased=unbiased, weight_mode=weight_mode)

    def std(self, unbiased=True, weight_mode='frequency'):
        # Note that this function does not account for weighting appropriately
        if self._count < 2:
            return np.nan
        return np.sqrt(self._compute_variance(unbiased=unbiased, weight_mode=weight_mode))
