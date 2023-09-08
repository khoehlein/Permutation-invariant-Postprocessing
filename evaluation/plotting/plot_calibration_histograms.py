import argparse
import os
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from data.config.interface import get_results_base_path
from evaluation.config.interface import get_paths
from evaluation.prediction_directory import SELECTED_RUNS_EUPP, PredictionStorage, \
    LegacyStorage, CSVReader
from experiments.baselines.bqn_utils import BernsteinQuantilePrediction, \
    EnsemblePrediction
from experiments.baselines.drn_utils import (
    LogisticPrediction, NormalPrediction, _LocScalePrediction
)
from utils.automation.storage import MultiRunExperiment
from utils.misc import get_timestamp_string


NUM_BINS = 20
NUM_MEMBERS = 10


def plot_calibration_histogram(ax, predictions, observations, num_bins):
    if isinstance(predictions, EnsemblePrediction):
        plot_upit_histogram(ax, predictions, observations, num_bins)
    elif isinstance(predictions, _LocScalePrediction):
        plot_pit_histogram(ax, predictions, observations, num_bins)
    else:
        raise NotImplementedError()


_histogram_kws = dict(
    density=True,
    edgecolor='#756fb3',
    facecolor='#b9b6d8',
)

def plot_rank_histogram(ax, y_pred: BernsteinQuantilePrediction, y_true: np.ndarray, num_bins: int):
    ranks = y_pred.compute_ranks(y_true, num_bins - 1)
    bins = np.linspace(0, num_bins, num_bins + 1)  + 0.5
    ax.axhline(1. / num_bins, linestyle='--', color='k')
    return ax.hist(ranks, bins, **_histogram_kws)


def plot_upit_histogram(ax, y_pred: EnsemblePrediction, y_true: np.ndarray, num_bins: int):
    upit = y_pred.compute_upit(y_true)
    bins = np.linspace(0, 1, num_bins + 1)
    ax.axhline(1., linestyle='--', color='k')
    return ax.hist(upit, bins, **_histogram_kws)


def plot_pit_histogram(ax, y_pred: _LocScalePrediction, y_true: np.ndarray, num_bins: int):
    pit = y_pred.compute_pit(y_true)
    bins = np.linspace(0, 1, num_bins + 1)
    ax.axhline(1., linestyle='--', color='k')
    return ax.hist(pit, bins, **_histogram_kws)


def plot_from_path(ax, path, data_fold, prediction_type: str = 'logistic', dataset='eupp'):
    experiment, run = MultiRunExperiment.from_run_path(path, _except_on_not_existing=True, return_run=True)

    prediction_dir = os.path.join(run.get_evaluation_path(), 'predictions', data_fold)
    npz_files = [f for f in sorted(os.listdir(prediction_dir)) if f.endswith('.npz')][:NUM_MEMBERS]
    y_pred = 0
    y_true = None
    for f in npz_files:
        data = np.load(os.path.join(prediction_dir, f))
        if y_true is None:
            y_true = data['obs']
        y_pred = y_pred + data['predictions']
    y_pred = y_pred / NUM_MEMBERS

    ensemble_size = {
        'test': NUM_BINS,
        'forecasts': NUM_BINS,
    }[data_fold]
    y_pred = cast_to_prediction(y_pred, prediction_type, ensemble_size)

    plot_calibration_histogram(ax, y_pred, y_true, ensemble_size + 1)


def plot_from_store(ax, store: Union[PredictionStorage, LegacyStorage, CSVReader], data_fold: str):
    y_pred = store.sample_ensemble(data_fold)
    if isinstance(y_pred, BernsteinQuantilePrediction):
        y_pred = y_pred.to_ensemble(NUM_BINS - 1)
    y_true = store.get_observations(data_fold)
    plot_calibration_histogram(ax, y_pred, y_true, NUM_BINS)


def get_prediction_store(model_name, dataset, lead_time):
    if dataset == 'eupp':
        return PredictionStorage.from_model_name(model_name, lead_time, dataset=dataset)
    elif dataset == 'gusts':
        if model_name.startswith('ST') or model_name.startswith('ED'):
            return PredictionStorage.from_model_name(model_name, lead_time, dataset=dataset)
        else:
            file_name = f'{get_results_base_path()}/gusts/schulz_lerch_original/complete/step{lead_time}.csv'
            if model_name == 'DRN':
                return CSVReader(file_name, LogisticPrediction)
            elif model_name == 'BQN':
                return CSVReader(file_name, BernsteinQuantilePrediction)
            else:
                raise ValueError(f'[ERROR] Unknownmodel type: {model_name}')


def cast_to_prediction(y_pred, prediction_type, ensemble_size):
    y_pred = {
        'logistic': LogisticPrediction,
        'normal': NormalPrediction,
        'bqn': BernsteinQuantilePrediction,
    }[prediction_type](y_pred)
    if isinstance(y_pred, BernsteinQuantilePrediction):
        y_pred = y_pred.to_ensemble(ensemble_size)
    return y_pred


def plot_tl_vs_tn():
    paths = get_paths('euppbench_tl_tn')

    def set_labels(axs):
        for j, label in enumerate([24, 72, 120]):
            axs[0, j].set(title=f'{label}h')
            axs[-1, j].set(xlabel='uPIT')
        for i, label in enumerate(order):
            axs[i, 0].set(ylabel=label)

    order = ['DRN-TL', 'DRN-TN']

    save_figures = True
    output_path = f'{get_results_base_path()}/eupp/figures'
    ts = get_timestamp_string()

    fig, axs = plt.subplots(2, 3, figsize=(10, 3.25), sharey='all', sharex='all')
    fig.suptitle('Reforecast dataset')
    for i, model_key in enumerate(order):
        pred_type = 'logistic' if 'TL' in model_key else 'normal'
        flt_paths = paths[model_key]
        for j, flt in enumerate(sorted(flt_paths.keys())):
            path = flt_paths[flt]
            plot_from_path(axs[i, j], path, 'test', prediction_type=pred_type)
    set_labels(axs)
    plt.tight_layout()

    if save_figures:
        plt.savefig(os.path.join(output_path, f'calibration_reforecasts_DRN-TL-vs-TN_{ts}.pdf'))
    plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 3, figsize=(10, 3.25), sharey='all', sharex='all')
    fig.suptitle('Forecast dataset')
    for i, model_key in enumerate(order):
        pred_type = 'logistic' if 'TL' in model_key else 'normal'
        flt_paths = paths[model_key]
        for j, flt in enumerate(sorted(flt_paths.keys())):
            path = flt_paths[flt]
            plot_from_path(axs[i, j], path, 'forecasts', prediction_type=pred_type)
    set_labels(axs)
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(output_path, f'calibration_forecasts_DRN-TL-vs-TN_{ts}.pdf'))
    plt.show()
    plt.close()


def plot_bqn_sum_vs_ens():
    paths = get_paths('euppbench_bqn_versions')
    paths['BQN-Ens'] = paths['BQN-Ens-Q']

    def set_labels(axs):
        for j, label in enumerate([24, 72, 120]):
            axs[0, j].set(title=f'{label}h')
            axs[-1, j].set(xlabel='uPIT')
        for i, label in enumerate(order):
            axs[i, 0].set(ylabel=label)

    order = ['BQN-Sum', 'BQN-Ens']

    save_figures = True
    output_path = f'{get_results_base_path()}/eupp/figures'
    ts = get_timestamp_string()

    fig, axs = plt.subplots(2, 3, figsize=(10, 3.25), sharey='all', sharex='all')
    fig.suptitle('Reforecast dataset')
    for i, model_key in enumerate(order):
        flt_paths = paths[model_key]
        for j, flt in enumerate(sorted(flt_paths.keys())):
            path = flt_paths[flt]
            plot_from_path(axs[i, j], path, 'test', prediction_type='bqn')
    set_labels(axs)
    plt.tight_layout()

    if save_figures:
        plt.savefig(os.path.join(output_path, f'calibration_reforecasts_BQN-Sum-vs-Ens_{ts}.pdf'))
    plt.show()
    plt.close()

    order = ['BQN-Sum', 'BQN-Ens-Q', 'BQN-Ens-R']

    fig, axs = plt.subplots(3, 3, figsize=(10, 4.25), sharey='all', sharex='all')
    fig.suptitle('Forecast dataset')
    for i, model_key in enumerate(order):
        flt_paths = paths[model_key]
        for j, flt in enumerate(sorted(flt_paths.keys())):
            path = flt_paths[flt]
            plot_from_path(axs[i, j], path, 'forecasts', prediction_type='bqn')
    set_labels(axs)
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(output_path, f'calibration_forecasts_BQN-Sum-vs-Ens_{ts}.pdf'))
    plt.show()
    plt.close()


def plots_for_eupp():
    drn_paths = SELECTED_RUNS_EUPP
    order = [
        'BQN', 'ED-BQN', 'ST-BQN',
        'DRN', 'ED-DRN', 'ST-DRN',
    ]

    def set_labels(axs):
        for j, label in enumerate([24, 72, 120]):
            axs[0, j].set(title=f'{label}h')
            axs[-1, j].set(xlabel='uPIT')
        for i, label in enumerate(order):
            axs[i, 0].set(ylabel=label)

    save_figures = True
    output_path = f'{get_results_base_path()}/wge/eupp/figures'
    ts = get_timestamp_string()

    fig, axs = plt.subplots(6, 3, figsize=(10, 8), sharey='all', sharex='all')
    fig.suptitle('EUPPBench reforecasts')
    for i, model_key in enumerate(order):
        pred_type = 'logistic' if 'DRN' in model_key else 'bqn'
        flt_paths = drn_paths[model_key]
        for j, flt in enumerate(sorted(flt_paths.keys())):
            path = flt_paths[flt]
            plot_from_path(axs[i, j], path, 'test', prediction_type=pred_type)
    set_labels(axs)
    plt.tight_layout()

    if save_figures:
        plt.savefig(os.path.join(output_path, f'calibration_reforecasts_eupp.pdf'))
    plt.show()

    plt.close()

    fig, axs = plt.subplots(6, 3, figsize=(10, 8), sharey='all', sharex='all')
    fig.suptitle('EUPPBench forecasts')
    for i, model_key in enumerate(order):
        pred_type = 'logistic' if 'DRN' in model_key else 'bqn'
        flt_paths = drn_paths[model_key]
        for j, flt in enumerate(sorted(flt_paths.keys())):
            path = flt_paths[flt]
            plot_from_path(axs[i, j], path, 'forecasts', prediction_type=pred_type)
    set_labels(axs)
    plt.tight_layout()
    if save_figures:
        plt.savefig(os.path.join(output_path, f'calibration_forecasts_eupp.pdf'))
    plt.show()
    plt.close()


def plots_for_gusts():
    order = [
        'BQN', 'ED-BQN', 'ST-BQN',
        'DRN', 'ED-DRN', 'ST-DRN',
    ]
    lead_times = [6, 12, 18]

    def set_labels(axs):
        for j, label in enumerate([6, 12, 18]):
            axs[0, j].set(title=f'{label}h')
            axs[-1, j].set(xlabel='uPIT')
        for i, label in enumerate(order):
            axs[i, 0].set(ylabel=label)

    save_figures = True
    output_path = f'{get_results_base_path()}/gusts/figures'
    ts = get_timestamp_string()

    fig, axs = plt.subplots(len(order), len(lead_times), figsize=(10, 8), sharey='all', sharex='all')
    fig.suptitle('Wind gust forecasts')
    for i, model_key in enumerate(order):
        for j, flt in enumerate(lead_times):
            store = get_prediction_store(model_key, 'gusts', flt)
            plot_from_store(axs[i, j], store, 'test')
    set_labels(axs)
    plt.tight_layout()

    if save_figures:
        plt.savefig(os.path.join(output_path, f'calibration_forecasts_gusts.pdf'))
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--posterior', type=str, required=True, choices=['bqn', 'logistic','normal'])
    parser.add_argument('--ensemble-size', type=int, required=True)
    args = vars(parser.parse_args())

    experiment = MultiRunExperiment(args['path'], _except_on_not_existing=True)

    def process_run(run):
        # experiment, run = MultiRunExperiment.from_run_path(args['path'], return_run=True, _except_on_not_existing=True)

        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        fig.suptitle(run.path)
        for i, mode in enumerate(['valid', 'test', 'forecasts']):
            ensemble_size = {
                'valid': 20,
                'test': 20,
                'forecasts': 20,
            }[mode]
            prediction_dir = os.path.join(run.get_evaluation_path(), 'predictions', mode)
            npz_files = [f for f in sorted(os.listdir(prediction_dir)) if f.endswith('.npz')][:NUM_MEMBERS]
            y_pred = 0
            y_true = None
            for f in npz_files:
                data = np.load(os.path.join(prediction_dir, f))
                if y_true is None:
                    y_true = data['obs']
                y_pred = y_pred + data['predictions']
            y_pred = y_pred / NUM_MEMBERS
            y_pred = cast_to_prediction(y_pred, args['posterior'], ensemble_size)
            plot_calibration_histogram(axs[i], y_pred, y_true, ensemble_size + 1)
            axs[i].set(title=mode)
        plt.tight_layout()
        plt.show()
        plt.close()

    experiment.map_across_runs(process_run)


if __name__ == '__main__':
    plots_for_eupp()
    plots_for_gusts()
    # plot_tl_vs_tn()
    # main()
    # plot_bqn_sum_vs_ens()
