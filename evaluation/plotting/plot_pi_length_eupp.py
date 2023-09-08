import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import tqdm
from scipy.stats import bootstrap

from data.config.interface import get_results_base_path
from evaluation.prediction_directory import PredictionStorage
from utils.automation.storage import MultiRunExperiment

ENSEMBLE_SIZE = 10
NUM_SAMPLES = 50
XLIM = [0.78, 1.02]


def compute_pi_length(predictions: list, cover_probs: np.ndarray):
    data = np.zeros((len(predictions), len(cover_probs)))
    with tqdm.tqdm(total=int(np.prod(data.shape))) as pbar:
        for i, y_pred in enumerate(predictions):
            for j, p in enumerate(cover_probs):
                data[i, j] = y_pred.compute_pi_length(alpha=(1.-p)).mean()
                pbar.update(1)
    return xr.DataArray(data, dims=['sample', 'p_ref'], coords={'p_ref': (['p_ref'], cover_probs)})


def _pil_boxplot(ax, pi_length: xr.DataArray):
    p_ref = pi_length.p_ref.values
    ax.boxplot(
        pi_length.values,
        positions=p_ref,
        widths=np.min(np.abs(np.diff(p_ref))) * 0.8)
    ylabel = 'PI length'
    xticks = np.round(p_ref, decimals=2)
    ax.set(
        xlabel='Nominal level', ylabel=ylabel,
        xlim=XLIM,
        xticks=xticks
    )
    ax.set_xticklabels(xticks, rotation=45)


def _pil_lineplot(ax, pi_length: xr.DataArray, plot_style_kws=None):
    p_ref = pi_length.p_ref.values
    if plot_style_kws is None:
        plot_style_kws = {}
    pil_mean = pi_length.mean(dim='sample').values
    # pil_q25 = pi_length.quantile(0.25, dim='sample').values
    # pil_q75 = pi_length.quantile(0.75, dim='sample').values
    lines = ax.plot(
        p_ref, pil_mean,
        # yerr=np.stack([pil_mean - pil_q25, pil_q75 - pil_mean], axis=0),
        **plot_style_kws
    )
    return lines


def _pil_difference_lineplot(ax, pi_length: xr.DataArray, pi_benchmark: xr.DataArray, plot_style_kws=None):
    p_ref = pi_length.p_ref.values
    if plot_style_kws is None:
        plot_style_kws = {}
    pil_ref = pi_benchmark.isel(sample=0)
    pil_difference = 100 * ((pi_length - pil_ref)/ pil_ref).transpose('sample', 'p_ref').values
    pil_mean = np.mean(pil_difference, axis=0)
    bounds = bootstrap([pil_difference], np.mean, method='basic').confidence_interval
    yerr = np.stack([bounds.low, bounds.high], axis=0) - pil_mean
    # pil_q25 = pi_length.quantile(0.25, dim='sample').values
    # pil_q75 = pi_length.quantile(0.75, dim='sample').values
    lines = ax.plot(
        p_ref, pil_mean,
        # yerr=np.stack([pil_mean - pil_q25, pil_q75 - pil_mean], axis=0),
        **plot_style_kws
    )
    c = plot_style_kws['color'] if 'color' in plot_style_kws else None
    ax.fill_between(
        p_ref, bounds.low, bounds.high, color=lines[0].get_color(),
        alpha=0.2
    )
    return lines


def draw_pil_boxplot(ax, predictions: list, coverage_probs: np.array):
    pi_length = compute_pi_length(predictions, coverage_probs)
    _pil_boxplot(ax, pi_length)


def get_reference_proabilities(ensemble_size: int):
    # return np.arange(ensemble_size - 1, 1, -2) / (ensemble_size + 1)
    return np.arange(0.8, 1., 0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['logistic', 'normal', 'bqn'])
    parser.add_argument('--data-fold', type=str, default='test', choices=['test', 'valid', 'forecasts'])
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--num-members', type=int, required=True)
    args = vars(parser.parse_args())
    path = args['path']
    mode = args['mode']
    dataset = args['data_fold']

    print(f'[INFO] Evaluating {path}')
    experiment = MultiRunExperiment(path)
    params = pd.DataFrame(experiment.list_run_parameters())
    params = params.drop([
        c for c in params.columns
        if
        (c not in {'run_name', 'time_stamp'}) and (np.all(params[c] == params[c].iloc[0]) or np.all(params[c].isnull()))
    ], axis=1)
    cover_probs = get_reference_proabilities(args['num_members'])
    for i in range(len(params)):
        print('[INFO] Parameters:')
        print(params.iloc[i])
        run = experiment.load_run(params['run_name'].iloc[i])
        prediction_storage = PredictionStorage.from_run_path(run.path, mode)
        predictions = [
            prediction_storage.sample_ensemble(dataset, ensemble_size=ENSEMBLE_SIZE)
            for _ in range(NUM_SAMPLES)
        ]

        fig, ax = plt.subplots(1, 1, dpi=300, figsize=(8, 4))
        fig.suptitle(f'{path} ({dataset})')
        draw_pil_boxplot(ax, predictions, cover_probs)
        plt.tight_layout()
        plt.show()
        plt.close()


def _get_benchmarks(flt, dataset, reference_probs):
    drn_benchmark = PredictionStorage.from_model_name('DRN', flt)
    pil_drn_benchmark = compute_pi_length([
        drn_benchmark.sample_ensemble(dataset, ensemble_size=ENSEMBLE_SIZE)
        for _ in range(NUM_SAMPLES)
    ], reference_probs)
    bqn_benchmark = PredictionStorage.from_model_name('BQN', flt)
    pil_bqn_benchmark = compute_pi_length([
        bqn_benchmark.sample_ensemble(dataset, ensemble_size=ENSEMBLE_SIZE)
        for _ in range(NUM_SAMPLES)
    ], reference_probs)
    return {'DRN': pil_drn_benchmark, 'BQN': pil_bqn_benchmark}


def _draw_length_difference_plot(ax, flt, dataset, benchmarks, labels=True):
    drn_names = ['ED-DRN', 'ST-DRN']
    bqn_names = ['ED-BQN', 'ST-BQN']
    for j, (drn_name, bqn_name) in enumerate(zip(drn_names, bqn_names)):
        drn_storage = PredictionStorage.from_model_name(drn_name, flt)
        pil_drn = compute_pi_length([
            drn_storage.sample_ensemble(dataset, ensemble_size=ENSEMBLE_SIZE)
            for _ in range(NUM_SAMPLES)
        ], benchmarks['DRN']['p_ref'].values)
        lines = _pil_difference_lineplot(
            ax, pil_drn, benchmarks['DRN'],
            plot_style_kws={'linestyle': '-', 'label': drn_name if labels else None, 'alpha': 0.8}
        )
        bqn_storage = PredictionStorage.from_model_name(bqn_name, flt)
        pil_bqn = compute_pi_length([
            bqn_storage.sample_ensemble('test', ensemble_size=ENSEMBLE_SIZE)
            for _ in range(NUM_SAMPLES)
        ], benchmarks['BQN']['p_ref'].values)
        lines = _pil_difference_lineplot(
            ax, pil_bqn, benchmarks['BQN'],
            plot_style_kws={'linestyle': '--', 'label': bqn_name if labels else None, 'color': lines[0].get_color(), 'alpha': 0.8})


def _draw_difference_plots_for_all_leadtimes(axs, dataset, lead_times, reference_ensemble_size, labels=True):
    reference_probabilities = get_reference_proabilities(reference_ensemble_size)
    for i, flt in enumerate(lead_times):
        benchmarks = _get_benchmarks(flt, dataset, reference_probabilities)
        _draw_length_difference_plot(axs[i], flt, dataset, benchmarks, labels=labels)
        axs[i].grid(alpha=0.5)
        axs[i].axvline((reference_ensemble_size - 1) / (reference_ensemble_size + 1), linestyle='--', color='k',
                       alpha=0.5, zorder=0)
        axs[i].set(yscale='linear', xlim=XLIM)


def plot_pi_length_difference(save_figures=False):
    lead_times = [24, 72, 120]
    output_path = f'{get_results_base_path()}/eupp/figures'
    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=300, sharex='all', sharey='row')
        axs[0, 0].set(ylabel='PIL diff. (reforecasts) [%]')
        axs[1, 0].set(ylabel='PIL diff. (forecasts) [%]')
        axs[1, 1].set(xlabel='Nominal level')
        _draw_difference_plots_for_all_leadtimes(axs[0], 'test', lead_times, 11, labels=True)
        _draw_difference_plots_for_all_leadtimes(axs[1], 'forecasts', lead_times, 51, labels=True)
        for i, flt in enumerate(lead_times):
            axs[0, i].set(title=f'{flt}h', yscale='linear', xlim=XLIM)
        axs[0, 2].legend(loc='upper right')
        plt.tight_layout()
        if save_figures:
            plt.savefig(os.path.join(output_path, f'pil_difference_joint_eupp.pdf'))
        plt.show()
        plt.close()


def plot_pi_length_for_paper(save_figures=False):
    drn_names = ['ED-DRN', 'ST-DRN', 'DRN']
    bqn_names = ['ED-BQN', 'ST-BQN', 'BQN']
    flts = [24, 72, 120]
    output_path = f'{get_results_base_path()}/eupp/figures'
    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=300, sharex='all', sharey='row')
        ylabel = 'PI length [K]'
        axs[0, 0].set(ylabel='PI length (reforecasts) [K]')
        axs[1, 0].set(ylabel='PI length (forecasts) [K]')
        axs[1, 1].set(xlabel='Nominal level')
        for i, flt in enumerate(flts):
            axs[0, i].grid(alpha=0.5)
            axs[0, i].axvline(10 / 12, linestyle='--', color='k', alpha=0.5, zorder=0)
            axs[0, i].set(title=f'{flt}h', yscale='linear', xlim=XLIM)
            axs[1, i].grid(alpha=0.5)
            axs[1, i].axvline(50 / 52, linestyle='--', color='k', alpha=0.5, zorder=0)
            axs[1, i].set(yscale='linear', xlim=XLIM)
            for j, (drn_name, bqn_name) in enumerate(zip(drn_names, bqn_names)):
                drn_storage = PredictionStorage.from_model_name(drn_name, flt)
                pil_drn = compute_pi_length([
                    drn_storage.sample_ensemble('test', ensemble_size=ENSEMBLE_SIZE)
                    for _ in range(NUM_SAMPLES)
                ], get_reference_proabilities(11))
                lines = _pil_lineplot(axs[0, i], pil_drn, plot_style_kws={'linestyle': '-', 'label': drn_name, 'alpha': 0.8})
                bqn_storage = PredictionStorage.from_model_name(bqn_name, flt)
                pil_bqn = compute_pi_length([
                    bqn_storage.sample_ensemble('test', ensemble_size=ENSEMBLE_SIZE)
                    for _ in range(NUM_SAMPLES)
                ], get_reference_proabilities(11))
                lines = _pil_lineplot(axs[0, i], pil_bqn, plot_style_kws={'linestyle': '--', 'label': bqn_name, 'color': lines[0].get_color(), 'alpha': 0.8})
                # xticks = np.round(get_reference_proabilities(11), decimals=2)
                # axs[0, i].set(
                #     xlim=[-0.05, 1.05],
                #     xticks=xticks
                # )
                # axs[0, i].set_xticklabels(xticks)

                # axs[1, i].set(title=f'{flt}h (forecasts)')
                pil_drn = compute_pi_length([
                    drn_storage.sample_ensemble('forecasts', ensemble_size=ENSEMBLE_SIZE)
                    for _ in range(NUM_SAMPLES)
                ], get_reference_proabilities(51))
                lines = _pil_lineplot(axs[1, i], pil_drn, plot_style_kws={'linestyle': '-', 'label': drn_name, 'alpha': 0.8})
                pil_bqn = compute_pi_length([
                    bqn_storage.sample_ensemble('forecasts', ensemble_size=ENSEMBLE_SIZE)
                    for _ in range(NUM_SAMPLES)
                ], get_reference_proabilities(51))
                lines = _pil_lineplot(axs[1, i], pil_bqn, plot_style_kws={'linestyle': '--', 'label': bqn_name, 'color': lines[0].get_color(), 'alpha': 0.8})

                # xticks = np.round(get_reference_proabilities(51), decimals=2)
                # axs[1, i].set(
                #     xlim=[-0.05, 1.05],
                #     xticks=xticks
                # )
                # axs[1, i].set_xticklabels(xticks)

        axs[0,0].legend(loc='upper left')
        plt.tight_layout()
        if save_figures:
            plt.savefig(os.path.join(output_path, f'pil_joint_eupp.pdf'))
        plt.show()
        plt.close()


if __name__ == '__main__':
    plot_pi_length_for_paper(save_figures=True)
    plot_pi_length_difference(save_figures=True)
