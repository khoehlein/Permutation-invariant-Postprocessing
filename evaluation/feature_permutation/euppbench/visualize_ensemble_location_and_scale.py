import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from evaluation.feature_permutation.perturbations import Ranking


def plot_statistics_relations(data: np.ndarray, label):
    loc_metrics = dict([
        ('mean', np.mean(data, axis=-1)),
        ('median', np.median(data, axis=-1)),
        # ('min', np.min(data, axis=-1)),
        # ('max', np.max(data, axis=-1))
    ])
    scale_metrics = dict([
        ('std', np.std(data, axis=-1, ddof=1)),
        ('iqr', np.quantile(data, 0.75, axis=-1) - np.quantile(data, 0.25, axis=-1)),
        # ('range', loc_metrics['max'] - loc_metrics['min'])
    ])
    sorted = np.sort(data, axis=-1)

    summary = []

    fig, axs = plt.subplots(len(loc_metrics), len(scale_metrics), sharex='col', sharey='row', figsize=(3 * len(scale_metrics), 3 * len(loc_metrics)), dpi=300)
    fig.suptitle(label)
    for i, (loc_name, loc_values) in enumerate(loc_metrics.items()):
        axs[i, 0].set(ylabel=loc_name)
        for j, (scale_name, scale_values) in enumerate(scale_metrics.items()):
            if i == 0:
                axs[-1, j].set(xlabel=scale_name)
            predictors = np.stack([loc_values, scale_values], axis=-1)
            model = LinearRegression()
            model.fit(predictors, sorted)
            score = model.score(predictors, sorted)
            axs[i, j].set(title=f'R^2 = {score}', xscale='log')
            axs[i, j].scatter(scale_values, loc_values, alpha=0.05)
            summary.append({
                'loc_metric': loc_name,
                'scale_metric': scale_name,
                'score': score
            })
    plt.tight_layout()
    plt.show()
    plt.close()
    return pd.DataFrame(summary)


def plot_pca_modes(data: np.ndarray, label: str, k=3):
    pca = PCA(whiten=False)
    pca.fit((data - np.mean(data, axis=0, keepdims=True)) / np.std(data, axis=0, keepdims=True))
    components = pca.components_

    fig = plt.figure(figsize=(8, 4), dpi=300)
    fig.suptitle(label)
    ax = fig.add_subplot(121)
    ax.scatter(np.arange(data.shape[-1]), pca.explained_variance_ratio_)
    ax.set(yscale='log')
    for i in range(k):
        ax = fig.add_subplot(k * 100 + 22 + 2 * i)
        ax.bar(np.arange(data.shape[-1]), components[i])
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_mean_vs_std(data: np.ndarray, yday, label: str):
    mu = np.mean(data, axis=-1)
    sigma = np.std(data, axis=-1, ddof=1)
    print(yday.min(), yday.max())
    fig, axs = plt.subplots(2, 2, dpi=300)
    fig.suptitle(label)
    ax = axs[0, 0]
    ax.scatter(yday,mu)
    # p = ax.scatter(mu, sigma, c=yday, alpha=0.1, cmap='hsv', vmin=0., vmax=1.)
    ax.set(xlabel='yday', ylabel='mean')
    ax = axs[1, 0]
    ax.scatter(yday, sigma)
    # p = ax.scatter(mu, sigma, c=yday, alpha=0.1, cmap='hsv', vmin=0., vmax=1.)
    ax.set(xlabel='yday', ylabel='std')
    # plt.colorbar(p, ax=ax)
    plt.show()


def plot_bin_occupation(data: np.ndarray, label: str, grouper: str):
    mu = np.mean(data, axis=-1)
    ranks_mu = Ranking(mu, method='unique')
    sigma = np.std(data, axis=-1, ddof=1)
    ranks_sigma = Ranking(sigma, method='unique')
    num_bins = [5, 10, 20, 50, 100, 200, 500, 1000]

    def _plot_stats(n: int, axs):
        bin_mu = ranks_mu.get_bins(n)
        bin_sigma = ranks_sigma.get_bins(n)
        df = pd.DataFrame({
            'bin_mu': bin_mu,
            'bin_sigma': bin_sigma,
            'mu': mu,
            'sigma': sigma,
        })
        grouped_mu = df.groupby(by=grouper)
        count_ = grouped_mu.count()
        mean_ = grouped_mu.mean()
        min_ = grouped_mu.min()
        max_ = grouped_mu.max()
        median_ = grouped_mu.median()
        q25_ = grouped_mu.quantile(0.25)
        q75_ = grouped_mu.quantile(0.75)

        x = mean_['mu'].index.values + 0.5
        ax = axs[0]
        ax.bar(x, count_['mu'], color='k', width=1.)
        ax.set(yscale='log', title=f', N = {n}')
        ax = axs[1]
        ax.plot(x, min_['mu'].values, color='k', linestyle=':')
        ax.plot(x, max_['mu'].values, color='k', linestyle=':')
        ax.plot(x, q25_['mu'].values, color='k', linestyle='--')
        ax.plot(x, q75_['mu'].values, color='k', linestyle='--')
        ax.plot(x, median_['mu'].values, color='r', linestyle='-')
        ax.plot(x, mean_['mu'].values, color='b', linestyle='-')
        ax = axs[2]
        ax.plot(x, min_['sigma'].values, color='k', linestyle=':')
        ax.plot(x, max_['sigma'].values, color='k', linestyle=':')
        ax.plot(x, q25_['sigma'].values, color='k', linestyle='--')
        ax.plot(x, q75_['sigma'].values, color='k', linestyle='--')
        ax.plot(x, median_['sigma'].values, color='r', linestyle='-')
        ax.plot(x, mean_['sigma'].values, color='b', linestyle='-')
        ax.set(yscale='log')



    fig, axs = plt.subplots(3, len(num_bins), figsize=(2 * len(num_bins), 6), sharex='col',sharey='row',gridspec_kw={'height_ratios': [1, 2, 2]})
    axs[0, 0].set(ylabel='count')
    axs[1, 0].set(ylabel='mean')
    axs[2, 0].set(ylabel='std')
    fig.suptitle(f'{label}, grouped by {grouper}')
    for i, n in enumerate(num_bins):
        _plot_stats(n, axs[:, i])
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    from data.euppbench.reforecasts import (
        DataConfig, build_training_dataset, DEFAULT_SPLIT
    )
    args = {
        'data:flt': 72,
        'data:target': 't2m',
        'data:predictors': None,
        'data:splitting': DEFAULT_SPLIT,
        'data:target_completeness': 0.5,
        'data:location_embedding': 'spherical'
    }
    labels = DataConfig.from_args(args)._predictor_names()
    data, conditions = build_training_dataset(args, torch.device('cpu'), test=True)
    data = data[-1]
    ensemble = data.tensors[0].data.cpu().numpy()
    yday = data.tensors[1].data.cpu().numpy()
    data = ensemble
    print(data.shape)

    for i,label in enumerate(labels):
        if label == 'ssrd6':
            plot_bin_occupation(data[..., i], label, 'bin_mu')
            plot_mean_vs_std(data[..., i], yday, label)

    for i, label in enumerate(labels):
        if label == 'ssrd6':
            plot_bin_occupation(data[..., i], label, 'bin_sigma')
        # plot_pca_modes(data[..., i], label)
        # summary = plot_statistics_relations(data[..., i], label)
        # print(f'[INFO] {label}:')
        # print(summary.sort_values(by='score'))


def main_gusts():
    from data.cosmo_de import (
        DEFAULT_SPLIT, DataConfig, build_training_dataset
    )
    args = {
        'data:flt': 18,
        'data:target': 'wind_speed_of_gust',
        'data:predictors': None,
        'data:splitting': DEFAULT_SPLIT,
        'data:location_embedding': 'spherical'
    }
    labels = DataConfig.from_args(args)._predictor_names()
    data, conditions = build_training_dataset(args, torch.device('cpu'), test=True)
    data = data[-1]
    data = data.tensors[0].data.cpu().numpy()
    print(data.shape)

    for i,label in enumerate(labels):
        if label != 'ASOB_S':
            continue
        plot_bin_occupation(data[..., i], label, 'bin_mu')

    for i, label in enumerate(labels):
        if label != 'ASOB_S':
            continue
        plot_bin_occupation(data[..., i], label, 'bin_sigma')
        # plot_pca_modes(data[..., i], label)
        # summary = plot_statistics_relations(data[..., i], label)
        # print(f'[INFO] {label}:')
        # print(summary.sort_values(by='score'))


if __name__ == '__main__':
    main()
    # main_gusts()
