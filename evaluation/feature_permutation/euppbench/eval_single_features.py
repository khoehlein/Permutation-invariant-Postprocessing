import gc
import os
import sys
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
import tqdm
from matplotlib import pyplot as plt
from networkx import rescale_layout
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, rankdata

from data.config.interface import get_results_base_path
from evaluation.feature_permutation.perturbations import MetricPreservationShuffle, EnsembleRankShuffle
from evaluation.prediction_directory import PredictionStorage
from data.euppbench.reforecasts import DataConfig as EUPPConfig
from data.cosmo_de import DataConfig as GustsConfig
from experiments.baselines.bqn_utils import BernsteinQuantilePrediction

from utils.progress import WelfordStatisticsTracker
OVERWRITE = False
OUTPUT_PATH = get_results_base_path() + '/{}/figures'

def get_channel_labels(args, dataset):
    config_class = {
        'eupp': EUPPConfig,
        'gusts': GustsConfig,
    }[dataset]
    config = config_class.from_args(args)
    return config._predictor_names()


def get_prediction(store: PredictionStorage, model_id: int):
    return store.get_prediction('test', model_id)


def get_observations(store: PredictionStorage):
    return store.get_observations('test')


def compute_crps(prediction, observations, return_mean=True):
    if isinstance(prediction, BernsteinQuantilePrediction):
        crps = prediction.compute_crps(observations, 66)
    else:
        crps = prediction.compute_crps(observations).mean()
    if return_mean:
        return crps.mean()
    return crps


def push_primary_to_front(channels, dataset):
    primary = {'eupp': 't2m', 'gusts': 'VMAX_10M'}[dataset]
    return [primary, *[c for c in channels if c != primary]]


def get_crps_data(dataset: str, model_name: str, flt: int, model_id: int = None):
    if model_id is not None:
        model_id = [model_id]
    else:
        model_id = range(20)
    shuffle_data = []
    channel_labels = None
    for model_id_ in model_id:
        data_for_id = _get_crps_data(dataset, model_name, flt, model_id_)
        if channel_labels is None:
            channel_labels = data_for_id['channel_labels']
            channel_labels = push_primary_to_front(channel_labels, dataset)
        shuffle_data_ = data_for_id['shuffled']
        shuffle_data_['baseline'] = [data_for_id['baseline']] * len(shuffle_data_)
        shuffle_data_['model_id'] = [model_id_] * len(shuffle_data_)
        shuffle_data.append(shuffle_data_)
    shuffle_data = pd.concat(shuffle_data, axis=0, ignore_index=True)
    shuffle_data = shuffle_data.set_index(['channel', 'model_id'])
    shuffle_data = shuffle_data.to_xarray()
    return shuffle_data, channel_labels


def _get_crps_data(dataset: str, model_name: str, flt: int, model_id: int):
    reference = PredictionStorage.from_model_name(model_name, flt, dataset=dataset)
    cache_dir = os.path.join(reference.get_cache_dir(), 'crps_perturbed_single_feature')
    os.makedirs(cache_dir, exist_ok=True)
    y_ref = get_prediction(reference, model_id)
    observations = get_observations(reference)
    crps_ref = compute_crps(y_ref, observations)
    args = reference.run.parameters()
    channel_labels = get_channel_labels(args, dataset)
    num_channels = len(channel_labels)
    cache_file_name = f'{model_name}_{flt}_{model_id}.csv'
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    if os.path.exists(cache_file_path) and not OVERWRITE:
        shuffle_data = pd.read_csv(cache_file_path, index_col=0)
    else:
        print(f'[INFO] Computing: {model_name}-{flt}h[{model_id}]')
        crps_shuffled = {'channel': channel_labels}

        def _compute_crps(perturbation, target):
            store = PredictionStorage.from_model_name(model_name, flt, perturbation=perturbation.name, dataset=dataset)
            y = get_prediction(store, model_id)
            _crps = compute_crps(y, observations)
            if target in crps_shuffled:
                crps_shuffled[target].append(_crps)
            else:
                crps_shuffled[target] = [_crps]

        with tqdm.tqdm(total=num_channels, file=sys.stdout) as pbar:
            for i, label in enumerate(channel_labels):
                _compute_crps(
                    MetricPreservationShuffle(
                        num_channels, [i],
                        num_bins=1, metric='mean',
                        preserve_ranking=False
                    ),
                    'random'
                )
                _compute_crps(
                    MetricPreservationShuffle(
                        num_channels, [i],
                        num_bins=1, metric='mean',
                        preserve_ranking=True
                    ),
                    'random_ranked'
                )
                _compute_crps(
                    EnsembleRankShuffle(num_channels, [i]),
                    'rank'
                )
                for num_bins in [100]:
                    for metric_name in MetricPreservationShuffle.available_metrics():
                        for ranked in [True]:
                            crps_name = f'{metric_name}{num_bins}'
                            if ranked:
                                crps_name = crps_name + '_ranked'
                            _compute_crps(
                                MetricPreservationShuffle(
                                    num_channels, [i],
                                    num_bins=num_bins, metric=metric_name                            ),
                                crps_name
                            )
                pbar.update(1)
        shuffle_data = pd.DataFrame(crps_shuffled)
        shuffle_data.to_csv(cache_file_path)
    return {
        'shuffled': shuffle_data,
        'baseline': crps_ref,
        'channel_labels': channel_labels,
    }


def compute_all(dataset: str):
    model_names = ['ED-DRN', 'ST-DRN', 'ED-BQN', 'ST-BQN']
    lead_times = {'eupp': [24, 72, 120], 'gusts': [6, 12, 18]}[dataset]
    for model_name in model_names:
        for flt in lead_times:
            for model_id in range(20):
                data = _get_crps_data(dataset, model_name, flt, model_id)
                del data
                gc.collect()


def compute_importance(data: xr.Dataset, perturbation_key: str, reference_key: str):
    crps = data[perturbation_key]
    crps_ref = data[reference_key]
    return (crps - crps_ref) / crps_ref


def set_channel_labels(ax, channel_labels):
    num_channels = len(channel_labels)
    ax.set(xticks=np.arange(num_channels))
    ax.set_xticklabels(channel_labels, rotation=90)


def plot_feature_importance(dataset: str, lead_times: List[int], perturbation_key: str, baseline_key: str, show_zero_reference=False, title=None, export_plots=False):
    if title is None:
        title = f'{perturbation_key} vs. {baseline_key}'
    model_names = ['ED-DRN', 'ED-BQN', 'ST-DRN', 'ST-BQN']
    num_models = len(model_names)
    offsets = np.arange(num_models)
    offsets = 0.8 * (offsets - np.mean(offsets)) / num_models
    width = 0.8 / num_models
    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(3, 1, sharex='all', figsize=(10, 6), dpi=300)
        # fig.suptitle(title)
        labels = None
        for j, flt in enumerate(lead_times):
            for i, model_name in enumerate(model_names):
                data, labels = get_crps_data(dataset, model_name, flt)
                importance = compute_importance(data, perturbation_key, baseline_key).sel(channel=labels).values
                positions = np.arange(len(labels)) + offsets[i]
                bars = axs[j].bar(
                    positions,
                    np.median(importance, axis=-1),
                    yerr=np.std(importance, axis=-1, ddof=1),
                    width=width*0.6, label=(model_name if j==0 else None),
                    alpha=0.6,
                    zorder=10,
                )
                axs[j].boxplot(
                    importance.T,
                    positions=positions,
                    widths=width * 0.8, showfliers=False,
                    medianprops={'color': bars.patches[0].get_facecolor(), 'alpha': 1.},
                    zorder=20
                )
                if show_zero_reference:
                    axs[j].axhline(0., linestyle='--', alpha=0.5, color='k')
                axs[j].grid(axis='x', alpha=0.5, zorder=0)
                axs[j].set(ylabel=f'Rel. Imp. ({flt}h)', yscale='linear')
        set_channel_labels(axs[-1], channel_labels=labels)
        plt.figlegend(framealpha=1.)
        plt.tight_layout()
        if export_plots:
            plt.savefig(os.path.join(OUTPUT_PATH.format(dataset), 'feature_importance.pdf'))
        plt.show()
        plt.close()


def plot_feature_importance_v2(dataset: str, lead_times: List[int], perturbation_key: str, baseline_key: str, show_zero_reference=False, title=None, export_plots=False):
    if title is None:
        title = f'{perturbation_key} vs. {baseline_key}'
    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(3, 1, sharex='all', figsize=(10, 6), dpi=300)
        # fig.suptitle(title)
        _draw_importance_plots(axs, dataset, lead_times, baseline_key, perturbation_key, show_zero_reference)
        plt.figlegend(framealpha=1.)
        plt.tight_layout()
        if export_plots:
            plt.savefig(os.path.join(OUTPUT_PATH.format(dataset), 'feature_importance.pdf'))
        plt.show()
        plt.close()


def plot_feature_importance_v3(
        perturbation_key: str, baseline_key: str,
        show_zero_reference=False, title=None,
        export_plots=False,
        labels_eupp=None,
        labels_gusts=None,
):
    channels_eupp = 28 if labels_eupp is None else len(labels_eupp)
    channels_gusts = 61 if labels_gusts is None else len(labels_gusts)
    if title is None:
        title = f'{perturbation_key} vs. {baseline_key}'
    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(3, 2, sharex='col', sharey='col', figsize=(20, 6), dpi=300, gridspec_kw={'width_ratios': [channels_eupp,channels_gusts]})
        axs[0, 0].set(title='EUPPBench')
        axs[0, 1].set(title='Wind gusts')
        # fig.suptitle(title)
        _draw_importance_plots(axs[:, 0], 'eupp', [24, 72, 120], baseline_key, perturbation_key, show_zero_reference, labels=labels_eupp)
        plt.figlegend(framealpha=1.)
        _draw_importance_plots(axs[:, 1], 'gusts', [6, 12, 18], baseline_key, perturbation_key, show_zero_reference, labels=labels_gusts)
        plt.tight_layout()
        if export_plots:
            output_path = f'{get_results_base_path()}/figures_joint'
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, 'feature_importance_ensemble_selected.pdf'))
        plt.show()
        plt.close()


def _draw_importance_plots(axs, dataset, lead_times, baseline_key, perturbation_key, show_zero_reference, labels=None):
    model_names = ['ED-DRN', 'ED-BQN', 'ST-DRN', 'ST-BQN']
    num_models = len(model_names)
    offsets = np.arange(num_models)
    offsets = 0.8 * (offsets - np.mean(offsets)) / num_models
    width = 0.8 / num_models
    for j, flt in enumerate(lead_times):
        for i, model_name in enumerate(model_names):
            data, labels_ = get_crps_data(dataset, model_name, flt)
            if labels is None:
                labels = labels_
            importance = compute_importance(data, perturbation_key, baseline_key).sel(channel=labels).values
            positions = np.arange(len(labels)) + offsets[i]
            bars = axs[j].bar(
                positions,
                np.median(importance, axis=-1),
                yerr=np.std(importance, axis=-1, ddof=1),
                width=width * 0.6, label=(model_name if j == 0 else None),
                alpha=0.6,
                zorder=10,
            )
            axs[j].boxplot(
                importance.T,
                positions=positions,
                widths=width * 0.8, showfliers=False,
                medianprops={'color': bars.patches[0].get_facecolor(), 'alpha': 1.},
                zorder=20
            )
            if show_zero_reference:
                axs[j].axhline(0., linestyle='--', alpha=0.5, color='k')
            axs[j].grid(axis='x', alpha=0.5, zorder=0)
            axs[j].set(ylabel=f'Rel. Imp. ({flt}h)', yscale='linear', xlim=[-1, len(labels)])
        set_channel_labels(axs[-1], channel_labels=labels)


def get_benjamini_hochberg_probability(values: np.ndarray, alpha: float):
    values = np.sort(values)
    critical_values = rankdata(values, method='ordinal') * alpha / len(values)
    return np.max(values[values < critical_values])


def compute_spearman_correlation_with_permutation_test(data: np.ndarray, num_samples=1000, two_sided=True):
    triu_indices = np.triu_indices(len(data), k=1)
    sample_indices= np.arange(len(data))[:, None]
    result = spearmanr(data, axis=1)[0]
    reference = np.abs(result[triu_indices])
    reference_sign = np.sign(result[triu_indices])
    tracker = WelfordStatisticsTracker()
    with tqdm.tqdm(total=num_samples,file=sys.stdout) as pbar:
        for i in range(num_samples):
            permutation = np.argsort(np.random.random(data.shape), axis=-1)
            c = spearmanr(data[sample_indices, permutation], axis=1)[0][triu_indices]
            if two_sided:
                test = np.abs(c) > reference
            else:
                test = reference_sign * c > reference
            tracker.update(test.astype(int))
            pbar.update(1)
    pvalue = np.zeros_like(result)
    pvalue[triu_indices] = tracker.mean()
    pvalue = pvalue + pvalue.T
    # plt.figure()
    # p = (np.arange(100) + 0.5) / 100
    # p_obs = np.mean(tracker.mean()[:, None] < p[None, :], axis=0)
    # plt.plot(p, p, linestyle='--', color='k')
    # plt.scatter(p, p_obs, alpha=0.1)
    # plt.show()
    # plt.close()
    return result, pvalue


def plot_half_grid(ax, x):
    zeros = np.zeros_like(x)
    ones = np.ones_like(x) * len(x)
    # horizontal
    X = np.stack([zeros, x], axis=0)
    Y = np.stack([x, x], axis=0)
    ax.plot(X, Y, color='k', alpha=0.1, zorder=0, linewidth=1)
    # vertical
    X = np.stack([x, x], axis=0)
    Y = np.stack([x, ones], axis=0)
    ax.plot(X, Y, color='k', alpha=0.1, zorder=0, linewidth=1)


def my_bipartite_layout(labels, width=3., aspect=4/3, scale=1., center=0., align='vertical'):
    height = width / aspect
    offset = (width / 2, height / 2)

    top = [l + '_scale' for l in labels]
    bottom = [l + '_loc' for l in labels]
    nodes = list(top) + list(bottom)

    left_xs = np.repeat(0, len(top))
    right_xs = np.repeat(width, len(bottom))
    left_ys = np.linspace(0, height, len(top))
    right_ys = np.linspace(0, height, len(bottom))

    top_pos = np.column_stack([left_xs, left_ys]) - offset
    bottom_pos = np.column_stack([right_xs, right_ys]) - offset

    pos = np.concatenate([top_pos, bottom_pos])
    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos


def plot_importance_correlation(dataset: str, model_name: str, flt: int, export_plots=False):
    data, labels = get_crps_data(dataset, model_name, flt)
    data = data.sel(channel=labels)
    loc_importance = compute_importance(data, 'random_ranked', 'mean100_ranked')
    scale_importance = compute_importance(data, 'random_ranked', 'std100_ranked')
    joint_labels = [label + '_loc' for label in labels] + [label + '_scale' for label in labels]
    joint_importance = np.concatenate([loc_importance.values, scale_importance.values], axis=0)
    correlation = compute_spearman_correlation_with_permutation_test(joint_importance, num_samples=2000)
    dist = np.maximum((1. - correlation[0][:len(labels), :len(labels)]) / 2., (1. - correlation[0][len(labels):, len(labels):]) / 2.)
    Z_loc = linkage(squareform(dist), method='ward', optimal_ordering=True)
    order_loc = leaves_list(Z_loc)
    order = np.concatenate([order_loc, order_loc + len(labels)])
    mask = correlation[1][order, :][:, order] >= 0.05
    mask[np.triu_indices(len(joint_labels), k=1)] = True
    c = np.ma.MaskedArray(correlation[0][order, :][:, order], mask=mask)
    x = np.arange(len(order)) + 0.5
    # fig, axs = plt.subplots(1, 1, figsize=(20, 20), dpi=300)
    # fig.suptitle(f'{model_name}, {flt}h')
    # ax = axs
    # ax.pcolor(c, vmin=-1., vmax=1., cmap='coolwarm', zorder=10)
    # ax.axhline(len(labels), color='k', linestyle='--', xmin=0., xmax=0.5)
    # ax.axvline(len(labels), color='k', linestyle='--', ymin=0.5)
    # # ax.grid(axis='both', alpha=0.5, zorder=0)
    # plot_half_grid(ax, x)
    # ax.set_xticks(x)
    # ax.set_xticklabels(np.asarray(joint_labels)[order], rotation=90)
    # ax.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False)
    # ax.set_yticks(x)
    # ax.set_yticklabels(np.asarray(joint_labels)[order])
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # plt.tight_layout()
    # if export_plots:
    #     plt.savefig(os.path.join(OUTPUT_PATH.format(dataset), f'importance_correlations_model-{model_name.lower()}_flt-{flt}.pdf'))
    # else:
    #     plt.show()
    # plt.close()

    # Settings forplot layout:
    #

    def nudge(pos, x_shift=0.6):
        return {n: (x + x_shift, y) if n.endswith('loc') else (x - x_shift, y) for n, (x, y) in pos.items()}

    adjmat_bipartite = correlation[0].copy()
    adjmat_bipartite[mask] = 0.
    adjmat_bipartite[:len(labels), :len(labels)] = 0.
    adjmat_bipartite[len(labels):, len(labels):] = 0.
    # adjmat_bipartite = adjmat_bipartite[order, :][:, order]

    graph = nx.from_numpy_array(np.abs(adjmat_bipartite))
    graph = nx.relabel_nodes(graph, {i: label for i, label in enumerate(np.asarray(joint_labels)[order])})
    positions = my_bipartite_layout(labels, aspect=5)
    label_positions = nudge(positions)

    with plt.style.context('seaborn-colorblind'):
        plt.figure(figsize=(8, 16))
        ax = plt.gca()
        nx.draw_networkx_edges(
            graph, pos=positions,
            width=np.abs(adjmat_bipartite)[adjmat_bipartite != 0.] * 5,
            alpha=0.5,
            ax=ax,
        )
        nx.draw_networkx_nodes(graph, pos=positions)
        nx.draw_networkx_labels(graph, pos=label_positions, ax=ax)
        ax.set(xlim=[-2.15,2.15], title=f'{model_name}, {flt}h', ylim=[-0.21, 0.21])
        ax.axis('off')
        plt.tight_layout()
        if export_plots:
            plt.savefig(os.path.join(OUTPUT_PATH.format(dataset),
                                     f'importance_correlation_graph_model-{model_name.lower()}_flt-{flt}.pdf'))
        plt.show()
        plt.close()



def plot_ensemble_stats(dataset: str, flt: int, export_plots=False):
    all_data = []
    model_names = ['ED-DRN', 'ED-BQN', 'ST-DRN', 'ST-BQN']
    metrics = ['mean', 'max', 'min', 'std', 'range', 'iqr', 'skew', 'kurt']
    perturbations = [f'{m}100_ranked' for m in metrics]

    output_path = os.path.join(OUTPUT_PATH.format(dataset), 'ensemble_channels')
    os.makedirs(output_path, exist_ok=True)

    for model_name in model_names:
        data, channel_labels = get_crps_data(dataset, model_name, flt)
        reference = 'random_ranked'
        data_for_plot = np.stack(
            [
                ((data[p] - data[reference]) / data[reference]).values
                for p in perturbations
            ],
            axis=0
        )
        data_for_normalization = ((data['baseline'] - data['random_ranked']) / data['baseline']).values
        normalized = data_for_plot / data_for_normalization
        all_data.append(normalized)
    all_data = np.stack(all_data, axis=0) # shape: (model_name, perturbation, channel, model_id)

    metric_interactions = xr.open_dataset(f'{get_results_base_path()}/{dataset}/cache/perturbations/metric_interactions_{flt}h.nc')
    for i, channel_label in enumerate(channel_labels):
        interactions = metric_interactions['spearman'].sel(channel=channel_label, perturbation=metrics, reference=metrics).values
        with plt.style.context('seaborn-colorblind'):
            fig, axs = plt.subplots(2, 2, figsize=(5.5, 9), dpi=300, sharex='col', gridspec_kw={'height_ratios': [1, 1],'width_ratios': [15, 1], 'wspace': 0.05, 'hspace': 0.05})
            axs[0, 0].set(title=f'{flt}h: {channel_label}')
            x = np.arange(len(perturbations)) + 0.5
            offsets = np.arange(len(model_names)) * 0.8 / len(model_names)
            offsets = offsets - np.mean(offsets)
            ax = axs[1, 0]
            ax.axhline(0., linestyle='--', color='k')
            ax.axhline(1., linestyle='--', color='k')
            handles = []
            for j, model_name in enumerate(model_names):
                importance = all_data[j, :, i, :]
                positions = x + offsets[j]
                width= 0.8 / len(model_names)
                bars = ax.bar(
                    positions,
                    np.median(importance, axis=-1),
                    width=width * 0.6, label=model_name,
                    alpha=0.6,
                    zorder=10,
                )
                handles.append(bars)
                ax.boxplot(
                    importance.T,
                    positions=positions,
                    widths=width * 0.8, showfliers=False,
                    medianprops={'color': bars.patches[0].get_facecolor(), 'alpha': 1.},
                    zorder=20
                )
            ax.set(xticks=x, xlabel='Perturbation', ylabel='Fractional skill', ylim=[-1.05, 2.05])
            ax.set_xticklabels(metrics, rotation=90)
            ax = axs[0,0]
            p = ax.pcolor(interactions.T[::-1], vmin=-1., vmax=1., cmap='coolwarm')
            cbar = plt.colorbar(p, cax=axs[0, 1], label='Correlation', ticks=[-1, 0, 1], fraction=0.05)
            ax.set(yticks=x, yticklabels=metrics[::-1], ylabel='Original')
            axs[1, 1].remove() #.legend(handles=handles, labels=model_names)
            axs[1, 0].legend(handles=handles, labels=model_names, loc='lower left', ncols=1)
            plt.tight_layout()
            if export_plots:
                plt.savefig(os.path.join(output_path, f'ensemble_channel_channel-{channel_label}_flt-{flt}h.pdf'))
            # plt.show()
            plt.close()

    print('Done')


def plot_ensemble_stats_joint(dataset: str, flt: int, export_plots=False):
    all_data = []
    model_names = ['ED-DRN', 'ED-BQN', 'ST-DRN', 'ST-BQN']
    metrics = ['mean', 'max', 'min', 'std', 'range', 'iqr', 'skew', 'kurt']
    perturbations = [f'{m}100_ranked' for m in metrics]
    channels_for_plot = {
        'eupp': ['t2m', 'stl1', 'ssrd6', 'tcwv', 'sshf6', 'p10fg6'],
        'gusts': ['VMAX_10M', 'WIND850', 'T1000','T_G', 'ASOB_S', 'FI850'],
    }[dataset]

    output_path = os.path.join(OUTPUT_PATH.format(dataset), 'ensemble_channels')
    os.makedirs(output_path, exist_ok=True)

    for model_name in model_names:
        data, channel_labels = get_crps_data(dataset, model_name, flt)
        reference = 'random_ranked'
        data_for_plot = np.stack(
            [
                ((data[p] - data[reference]) / data[reference]).values
                for p in perturbations
            ],
            axis=0
        )
        data_for_normalization = ((data['baseline'] - data['random_ranked']) / data['baseline']).values
        normalized = data_for_plot / data_for_normalization
        all_data.append(normalized)
    all_data = np.stack(all_data, axis=0) # shape: (model_name, perturbation, channel, model_id)

    metric_interactions = xr.open_dataset(f'{get_results_base_path()}/{dataset}/cache/perturbations/metric_interactions_{flt}h.nc')
    with plt.style.context('seaborn-colorblind'):

        fig, axs = plt.subplots(
            2, len(channels_for_plot) + 1,
            figsize=(4 * len(channels_for_plot) + 1, 6.15), dpi=300, sharex='col',
            gridspec_kw={
                'height_ratios': [2, 1], 'width_ratios': ([15] * len(channels_for_plot)) + [1],
                'wspace': 0.05, 'hspace': 0.05
            }
        )
        fig.suptitle(f'{flt}h')
        for i, channel_label in enumerate(channels_for_plot):
            interactions = metric_interactions['spearman'].sel(channel=channel_label, perturbation=metrics, reference=metrics).values
            axs[0, i].set(title=channel_label)
            x = np.arange(len(perturbations)) + 0.5
            offsets = np.arange(len(model_names)) * 0.8 / len(model_names)
            offsets = offsets - np.mean(offsets)
            ax = axs[1, i]
            ax.axhline(0., linestyle='--', color='k')
            ax.axhline(1., linestyle='--', color='k')
            handles = []
            global_i = channel_labels.index(channel_label)

            for j, model_name in enumerate(model_names):
                importance = all_data[j, :, global_i, :]
                positions = x + offsets[j]
                width= 0.8 / len(model_names)
                bars = ax.bar(
                    positions,
                    np.median(importance, axis=-1),
                    width=width * 0.6, label=model_name,
                    alpha=0.6,
                    zorder=10,
                )
                handles.append(bars)
                ax.boxplot(
                    importance.T,
                    positions=positions,
                    widths=width * 0.8, showfliers=False,
                    medianprops={'color': bars.patches[0].get_facecolor(), 'alpha': 1.},
                    zorder=20
                )
            ax.set(xticks=x, xlabel='Perturbation', ylim=[-0.3, 1.3])
            if i > 0:
                ax.set(yticklabels=[])
            ax.set_xticklabels(metrics, rotation=90)
            ax = axs[0, i]
            p = ax.pcolor(interactions.T[::-1], vmin=-1., vmax=1., cmap='coolwarm')
            cbar = plt.colorbar(p, cax=axs[0, -1], label='Correlation', ticks=[-1, 0, 1], fraction=0.05)
            ax.set(yticks=x, yticklabels=[])
        axs[0, 0].set(ylabel='Original metric', yticks=x, yticklabels=metrics[::-1])
        axs[1, 0].set(ylabel='Fractional skill')
        axs[1, -1].remove() #.legend(handles=handles, labels=model_names)
        legend_index = {
            'eupp': 0, 'gusts':2
        }[dataset]
        axs[1, legend_index].legend(handles=handles, labels=model_names, loc='upper right', ncols=1)
        plt.tight_layout()
        if export_plots:
            plt.savefig(os.path.join(output_path, f'ensemble_channel_joint_flt-{flt}h.pdf'))
        plt.show()
        plt.close()


def main():
    # plot_feature_importance('random', 'baseline')
    # plot_feature_importance('random_ranked', 'baseline')
    # plot_feature_importance('random', 'random_ranked', show_zero_reference=True)
    for perturbation_key in ['mean100_ranked', 'std100_ranked']:
        plot_feature_importance('random_ranked', perturbation_key)


def main2():
    dataset = 'gusts'
    model_names = ['ED-DRN', 'ED-BQN', 'ST-DRN', 'ST-BQN']
    lead_times = [6, 12, 18]
    for i, model_name in enumerate(model_names):
        for j, flt in enumerate(lead_times):
            plot_importance_correlation(dataset, model_name, flt, export_plots=True)


def main3():
    dataset = 'eupp'
    model_names = ['ED-DRN', 'ED-BQN', 'ST-DRN', 'ST-BQN']
    lead_times = [24, 72, 120]
    for i, model_name in enumerate(model_names):
        for j, flt in enumerate(lead_times):
            plot_importance_correlation(dataset, model_name, flt, export_plots=True)


def plots_for_gusts():
    dataset = 'gusts'
    lead_times = [6, 12, 18]
    export = True
    # plot_feature_importance(dataset, lead_times, 'random_ranked', 'baseline', show_zero_reference=True, title=None, export_plots=export)
    for flt in lead_times:
        plot_ensemble_stats_joint(dataset, flt, export_plots=export)
        # plot_ensemble_stats(dataset, flt, export_plots=export)


def plots_for_eupp():
    dataset = 'eupp'
    lead_times = [24, 72, 120]
    export = True
    # plot_feature_importance(dataset, lead_times, 'random_ranked', 'baseline', show_zero_reference=True, title=None, export_plots=export)
    for flt in lead_times:
        plot_ensemble_stats_joint(dataset, flt, export_plots=export)
        # plot_ensemble_stats(dataset, flt, export_plots=export)


def plot_feature_importance_selected(export_plots=False):
    labels_eupp = [
        't2m',
        'mx2t6', 'mn2t6',
        't850',
        'stl1', 'strd6',
        'tcw', 'tcwv',
        'u10', 'u100',
        'v10', 'v100',
        'tp6', 'cp6',
        'ssr6', 'ssrd6',
        'p10fg6'
    ]
    labels_gusts = [
        'VMAX_10M',
        'T_2M',  # 'T_2M_LS', 'T_2M_LS_S', 'T_2M_MS', 'T_2M_MS_S',
        'T1000', 'T950', 'T850',
        'T_G',  # 'T_G_LS', 'T_G_LS_S', 'T_G_MS', 'T_G_MS_S',
        'TD_2M',  # 'TD_2M_LS', 'TD_2M_LS_S', 'TD_2M_MS', 'TD_2M_MS_S',
        'WIND_10M', 'WIND950', 'WIND850',
        # 'U_10M', 'U950', 'U850',
        # 'V_10M', 'V950', 'V850',
        'RELHUM1000',
        # 'W_SO2',
        'ASOB_S',  # 'ASOB_S_LS', 'ASOB_S_LS_S', 'ASOB_S_MS', 'ASOB_S_MS_S',
        'FI1000', 'FI950', 'FI850', 'FI700', 'FI500',
    ]
    plot_feature_importance_v3('random_ranked', 'baseline', export_plots=export_plots, labels_gusts=labels_gusts,
                               labels_eupp=labels_eupp)


if __name__ == '__main__':
    # main2()
    # main3()
    # compute_all('gusts')
    # plots_for_eupp()
    # plots_for_gusts()
    plot_feature_importance_selected(export_plots=True)
