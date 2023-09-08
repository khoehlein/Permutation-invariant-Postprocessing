import gc
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

from data.config.interface import get_results_base_path
from evaluation.feature_permutation.euppbench.eval_single_features import get_prediction, get_observations, \
    compute_crps, compute_importance, set_channel_labels, OUTPUT_PATH
from evaluation.feature_permutation.perturbations import ScalarPredictorPermutation
from evaluation.prediction_directory import PredictionStorage
from experiments.euppbench_forecasts.predict_minimal import get_channel_labels

OVERWRITE = False

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
            channel_labels = data_for_id['channel_labels'][:-6]
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
    cache_dir = os.path.join(reference.get_cache_dir(), 'crps_perturbed_scalar_predictors')
    os.makedirs(cache_dir, exist_ok=True)
    y_ref = get_prediction(reference, model_id)
    observations = get_observations(reference)
    crps_ref = compute_crps(y_ref, observations)
    channel_labels = get_channel_labels(dataset)
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
                    ScalarPredictorPermutation(num_channels, [i]),
                    'random'
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
    model_names = ['DRN', 'BQN']
    lead_times = {'eupp': [24, 72, 120], 'gusts': [6, 12, 18]}[dataset]
    for model_name in model_names:
        for flt in lead_times:
            for model_id in range(20):
                data = _get_crps_data(dataset, model_name, flt, model_id)
                del data
                gc.collect()


def plot_feature_importance_v2(
        dataset: str, lead_times: List[int],
        perturbation_key: str, baseline_key: str,
        show_zero_reference=False, title=None,
        export_plots=False
):
    if title is None:
        title = f'{perturbation_key} vs. {baseline_key}'
    fig, axs = plt.subplots(3, 1, sharex='all', sharey='all', figsize=(10, 6), dpi=300)

    with plt.style.context('seaborn-colorblind'):
        # fig.suptitle(title)
        _draw_importance_plots(axs, dataset, lead_times, baseline_key, perturbation_key, show_zero_reference)
        plt.figlegend(framealpha=1.)
        plt.tight_layout()
        if export_plots:
            plt.savefig(os.path.join(OUTPUT_PATH.format(dataset), 'feature_importance_scalar.pdf'))
        plt.show()
        plt.close()


def plot_feature_importance_v3(
        perturbation_key: str, baseline_key: str,
        show_zero_reference=False, title=None,
        export_plots=False,
        labels_eupp=None,
        labels_gusts=None,
):
    if title is None:
        title = f'{perturbation_key} vs. {baseline_key}'
    channels_eupp = 29 if labels_eupp is None else len(labels_eupp)
    channels_gusts = 62 if labels_gusts is None else len(labels_gusts)
    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(3, 2, sharex='col', sharey='col', figsize=(20, 6.5), dpi=300, gridspec_kw={'width_ratios': [channels_eupp, channels_gusts]})
        axs[0, 0].set(title='EUPPBench')
        axs[0, 1].set(title='Wind gusts')
        # fig.suptitle(title)
        _draw_importance_plots(axs[:, 0], 'eupp', [24, 72, 120], baseline_key, perturbation_key, show_zero_reference, labels=labels_eupp)
        plt.figlegend(framealpha=1.)
        _draw_importance_plots(axs[:, 1], 'gusts', [6, 12, 18], baseline_key, perturbation_key, show_zero_reference, labels=labels_gusts)
        plt.tight_layout()
        if export_plots:
            output_path = f'{get_results_base_path()}/wge/figures_joint'
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, 'feature_importance_scalar_selected.pdf'))
        plt.show()
        plt.close()


def _draw_importance_plots(axs, dataset, lead_times, baseline_key, perturbation_key, show_zero_reference, labels=None):
    model_names = ['DRN', 'BQN']
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
                np.mean(importance, axis=-1),
                # yerr=np.std(importance, axis=-1, ddof=1),
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


def plot_feature_importance_selected(export_plots=False):
    labels_ens = ['ens_mean', 'ens_sd']
    labels_eupp = [
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
    labels_eupp = labels_ens + [s + '_mean' for s in labels_eupp]
    labels_gusts = [
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
    labels_gusts = labels_ens + [s + '_mean' for s in labels_gusts]
    plot_feature_importance_v3('random', 'baseline', export_plots=export_plots, labels_eupp=labels_eupp,
                               labels_gusts=labels_gusts)


if __name__ == '__main__':
    # compute_all('eupp')
    # compute_all('gusts')
    # plot_feature_importance_v2('eupp', [24, 72, 120], 'random', 'baseline')
    # plot_feature_importance_v2('gusts', [6, 12, 18], 'random', 'baseline')
    plot_feature_importance_selected(export_plots=True)
