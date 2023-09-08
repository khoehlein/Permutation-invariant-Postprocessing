import numpy as np
import pandas as pd
import scipy

from experiments.baselines.common import Coverage


def bisect(fn, y, iterations=25):
    x0 = np.zeros_like(y)
    x1 = np.ones_like(y)
    out = np.empty_like(y)
    y0 = fn(x0, x0 == 0.)
    below = y <= y0
    y1 = fn(x1, x1 == 1.)
    above = y >= y1
    out[below] = 0.
    out[above] = 1.
    compute = ~np.logical_or(below, above)
    if np.any(compute):
        x0 = x0[compute]
        x1 = x1[compute]
        y = y[compute]
        for _ in range(iterations):
            x_mid = (x0 + x1) / 2.
            r_mid = fn(x_mid, compute) - y
            smaller = r_mid <= 0
            x0[smaller] = x_mid[smaller]
            x1[~smaller] = x_mid[~smaller]
        out[compute] = (x0 + x1) / 2.
    return out


def _test_bisect():

    def fn(x):
        return x

    out = bisect(fn, np.array([0.7, 0.5, 0.2]))

    print(out)


def quickrank(data):
    order = np.argsort(data, axis=-1)
    ordered_data = data[np.arange(len(data))[:, None], order]
    diffs = np.diff(ordered_data, axis=-1)
    assert np.all(diffs > 0.)
    return np.argsort(order, axis=-1) + 1


def _test_quick_rank():
    ranks = quickrank(np.array([[1, 3, 2], [6, 2, 9], [4, 8, 1]])) + 1
    print(ranks)


class EnsemblePrediction(object):

    def __init__(self, data: np.ndarray):
        assert len(data.shape) == 2
        self.data = data

    @property
    def num_members(self):
        return self.data.shape[-1]

    def _compute_rankdata(self, observations: np.ndarray):
        assert len(observations.shape) in {1, 2}
        if len(observations.shape) == 1:
            observations = observations[:, None]
        assert observations.shape[-1] == 1
        data = np.concatenate([self.data, observations], axis=-1)
        ranks = quickrank(data)
        return ranks

    def compute_ranks(self, observations: np.ndarray):
        ranks = self._compute_rankdata(observations)
        return ranks[:, -1]

    @staticmethod
    def _ranks_to_upit(ranks):
        n = ranks.shape[-1]
        obs_rank = ranks[:, -1]
        multiplicity = np.sum((ranks == obs_rank[:, None]).astype(int), axis=-1)
        rand_min = obs_rank - multiplicity
        rand_max = obs_rank
        return (rand_min + (rand_max - rand_min) * np.random.rand(*rand_max.shape)) / n

    def compute_upit(self, observations: np.ndarray):
        ranks = self._compute_rankdata(observations)
        return self._ranks_to_upit(ranks)

    def compute_coverage(self, observations: np.ndarray, alpha: float = None):
        if alpha is None:
            alpha = 2 / (self.num_members + 1)
        ranks = self.compute_ranks(observations)
        coverage = Coverage(alpha).from_ranks(ranks, max_rank=(self.num_members + 1))
        return coverage

    def compute_pit(self, observations: np.ndarray):
        return self.compute_upit(observations)

    def compute_crps(self, observations: np.ndarray):
        # CRPS PWM from https: // link.springer.com / article / 10.1007 / s11004 - 017 - 9709 - 7
        assert len(observations.shape) in {1, 2}
        if len(observations.shape) == 1:
            observations = observations[:, None]
        deviation = np.mean(np.abs(self.data - observations), axis=-1)
        beta_0 = np.mean(self.data, axis=-1)
        beta_1 = np.mean(np.sort(self.data, axis=-1) * np.arange(self.num_members)[None, :]) / (self.num_members - 1)
        res = deviation + beta_0 - 2. * beta_1
        return pd.Series(data=res, name='CRPS')


class BernsteinQuantilePrediction(object):

    def __init__(self, data: np.ndarray):
        assert len(data.shape) == 2
        self.data = data

    @property
    def degree(self):
        return self.data.shape[-1] - 1

    def to_ensemble(self, num_samples: int):
        q_levels = np.arange(1, (num_samples + 1.)) / (num_samples + 1.)
        B = scipy.stats.binom.pmf(np.arange(self.degree + 1)[:, None], self.degree, q_levels[None, :])
        q = np.matmul(self.data, B)
        return EnsemblePrediction(q)

    def compute_ranks(self, observations: np.ndarray, num_samples: int):
        ensemble = self.to_ensemble(num_samples)
        return ensemble.compute_ranks(observations)

    def _quantile(self, weights: np.ndarray, p: np.ndarray):
        # weights.shape == (num_samples, self.degree + 1)
        # p.shape == (num_samples,)
        B = scipy.stats.binom.pmf(np.arange(self.degree + 1)[None, :], self.degree, p[:, None])
        q = np.sum(weights * B, axis=-1)
        return q

    def compute_pit(self, observations):

        def quantile(p: np.ndarray, mask):
            zeros = p <= 0.
            ones = p >= 1.
            compute = ~np.logical_or(zeros, ones)
            out = np.empty_like(p)
            out[zeros] = self.data[mask][zeros, 0]
            out[ones] = self.data[mask][ones, -1]
            if np.any(compute):
                out[compute] = self._quantile(self.data[mask][compute], p)
            return out

        out = bisect(quantile, observations)
        return out[0]

    def compute_upit(self, observations: np.ndarray, num_samples: int):
        ensemble = self.to_ensemble(num_samples)
        return ensemble.compute_upit(observations)

    def compute_crps(self, observations: np.ndarray, num_samples: int):
        ensemble = self.to_ensemble(num_samples)
        return ensemble.compute_crps(observations)

    def compute_coverage(self, observations: np.ndarray, alpha: float = 0.05):
        pit = self.compute_pit(observations)
        return Coverage(alpha).from_pit(pit)

    def compute_pi_length(self, alpha: float = 0.05):
        p_lower, p_upper = [
            np.full((len(self.data),), p)
            for p in [alpha / 2., 1. - alpha / 2.]
        ]
        return self._quantile(self.data, p_upper) - self._quantile(self.data, p_lower)

    # def compute_coverage(self, observations: np.ndarray, num_samples: int, alpha: float = None):
    #     ensemble = self.to_ensemble(num_samples)
    #     return ensemble.compute_coverage(observations, alpha=alpha)


if __name__ == '__main__':
    _test_quick_rank()