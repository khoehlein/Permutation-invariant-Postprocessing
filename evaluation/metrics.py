import os
import time

import numpy as np
import rpy2.robjects as ro
import scipy.special
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from scipy.stats import rankdata

pandas2ri.activate()


_r_base = importr('base')
_path_to_r_code = os.path.abspath(os.path.join(__file__, '../inference', '..', '..', 'external', 'pp_schulz_lerch'))
_r_source = ro.r['source']
_r_source(os.path.join(_path_to_r_code, 'fn_eval.R'))

_r_fn_scores_tlogis = ro.r['fn_scores_distr']
_r_fn_scores_tnorm = ro.r['fn_scores_distr_tnorm']
_r_fn_scores_ens = ro.r['fn_scores_ens']
_r_fn_cover = ro.r['fn_cover']
_r_fn_brier_score = ro.r['brier_score']


def quickrank(data: np.ndarray):
    order = np.argsort(data, axis=-1)
    num_samples = len(data)
    if np.all(np.diff(data[np.arange(num_samples)[:, None], order]) > 0):
        return np.argsort(order, axis=-1).astype(int) + 1
    return rankdata(data, method='ordinal', axis=-1)


def _quickrank_observations(ensemble, observations):
    data = np.concatenate([observations[:, None], ensemble], axis=-1)
    return quickrank(data)[:, 0]


def fn_scores_tlogis(f: np.ndarray, y: np.ndarray, n_ens=20):
    result = _r_fn_scores_tlogis(f, y, n_ens=n_ens)
    result = pandas2ri.rpy2py(result)
    return result


def fn_scores_tnorm(f: np.ndarray, y: np.ndarray, n_ens=20):
    result = _r_fn_scores_tnorm(f, y, n_ens=n_ens)
    result = pandas2ri.rpy2py(result)
    return result


def fn_scores_ens(ens: np.ndarray, y: np.ndarray, n_ens=20):
    result = _r_fn_scores_ens(ens, y, n_ens_ref=n_ens)
    result = pandas2ri.rpy2py(result)

    return result


def fn_cover(pit_or_ranks: np.ndarray, alpha=None, n_ens=20):
    if alpha is None:
        alpha = 2. / (n_ens + 1)
    result = _r_fn_cover(pit_or_ranks, alpha=alpha, n_ens=n_ens)
    return float(result)


def fn_brier_score(f: np.ndarray, y: np.ndarray, t: float = 0., distr: str = 'tlogis', t_distr: float = 0.):
    result = _r_fn_brier_score(f, y, t=t, distr=distr, t_distr=t_distr)
    return result


def compute_metrics_logistic(f: np.ndarray, y: np.ndarray, n_ens=20):
    scores = fn_scores_tlogis(f, y, n_ens=n_ens)
    cover = fn_cover(scores['pit'].values, n_ens=n_ens)
    # scores['brier'] = fn_brier_score(f, y)
    scores = scores.mean(axis=0)
    scores['cover'] = cover
    return scores


def compute_metrics_normal(f: np.ndarray, y: np.ndarray, n_ens=20):
    scores = fn_scores_tnorm(f, y, n_ens=n_ens)
    cover = fn_cover(scores['pit'].values, n_ens=n_ens)
    # scores['brier'] = fn_brier_score(f, y)
    scores = scores.mean(axis=0)
    scores['cover'] = cover
    return scores


def compute_metrics_bqn(alphas: np.ndarray, y: np.ndarray, n_ens=20, multiplier=6):
    # evaluation as shown in pp_bqn.R
    p_degree = alphas.shape[-1] - 1
    q_levels = np.arange(1, multiplier * (n_ens + 1.)) / (multiplier * (n_ens + 1.))
    B = scipy.stats.binom.pmf(np.arange(p_degree + 1)[:, None], p_degree, q_levels[None, :])
    q = np.matmul(alphas, B)
    scores = fn_scores_ens(q, y, n_ens=n_ens)
    # compute and correct rank according to pp_bqn.R
    # rank = _quickrank_observations(q, y)
    rank = np.ceil(_quickrank_observations(q, y) * (n_ens + 1) / (len(q_levels) + 1))
    # scores = scores.drop(['e_me', 'rank'], axis=1)
    scores['rank'] = rank
    scores['e_me'] = np.mean(alphas, axis=-1) - y
    cover = fn_cover(scores['rank'].values, n_ens=n_ens)
    scores = scores.mean(axis=0)
    scores['cover'] = cover
    return scores


def _test():
    f = np.exp(np.random.randn(100, 11))
    y = np.exp(np.random.randn(100))
    metrics = compute_metrics_bqn(f, y)
    bs = fn_brier_score(f, y)
    scores = fn_scores_tlogis(f, y)
    pit = scores['pit'].values
    cover = fn_cover(pit)
    metrics = compute_metrics_logistic(f, y)
    print('[INFO] Done')


if __name__== '__main__':
    _test()
