import os

import matplotlib.pyplot as plt

from data.config.interface import get_results_base_path
from evaluation.plotting.plot_pi_length_eupp import XLIM, compute_pi_length, ENSEMBLE_SIZE, NUM_SAMPLES, \
    get_reference_proabilities, _pil_lineplot, _pil_difference_lineplot
from evaluation.prediction_directory import LegacyStorage, CSVReader
from experiments.baselines.bqn_utils import BernsteinQuantilePrediction
from experiments.baselines.drn_utils import LogisticPrediction


def plot_pi_length_for_paper(save_figures=False):
    flts = [6, 12, 18]
    output_path = f'{get_results_base_path()}/gusts/figures'
    os.makedirs(output_path, exist_ok=True)

    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex='all', sharey='row')
        axs[0].set(ylabel='PI length (test) [m/s]')
        axs[1].set(xlabel='Nominal level')
        for i, flt in enumerate(flts):
            _plot_pil_for_ensemble_models(axs, flt, i)
            _plot_pil_for_minimal(axs, flt, i)
        axs[0].legend(loc='upper left')
        plt.tight_layout()
        if save_figures:
            plt.savefig(os.path.join(output_path, f'pil_joint_gusts.pdf'))
        plt.show()
        plt.close()


def plot_pil_difference(save_figures=False):
    flts = [6, 12, 18]
    output_path = f'{get_results_base_path()}/gusts/figures'
    os.makedirs(output_path, exist_ok=True)

    with plt.style.context('seaborn-colorblind'):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=300, sharex='all', sharey='row')
        axs[0].set(ylabel='PIL diff. (test) [%]')
        axs[1].set(xlabel='Nominal level')
        for i, flt in enumerate(flts):
            _plot_pil_difference_for_ensemble_models(axs, flt, i)
        axs[0].legend(loc='lower left')
        plt.tight_layout()
        if save_figures:
            plt.savefig(os.path.join(output_path, f'pil_difference_joint_gusts.pdf'))
        plt.show()
        plt.close()


def _plot_pil_for_ensemble_models(axs, flt, i, reference_ensemble_size=20):
    drn_names = ['ED-DRN', 'ST-DRN']
    bqn_names = ['ED-BQN', 'ST-BQN']
    axs[i].grid(alpha=0.5)
    axs[i].axvline((reference_ensemble_size - 1) / (reference_ensemble_size + 1), linestyle='--', color='k', alpha=0.5, zorder=0)
    axs[i].set(title=f'{flt}h', yscale='linear', xlim=XLIM)
    for j, (drn_name, bqn_name) in enumerate(zip(drn_names, bqn_names)):
        drn_storage = LegacyStorage.from_model_name(drn_name, flt)
        pil_drn = compute_pi_length([
            drn_storage.sample_ensemble('test', ensemble_size=ENSEMBLE_SIZE)
            for _ in range(NUM_SAMPLES)
        ], get_reference_proabilities(reference_ensemble_size))
        lines = _pil_lineplot(axs[i], pil_drn, plot_style_kws={'linestyle': '-', 'label': drn_name, 'alpha': 0.8})

        bqn_storage = LegacyStorage.from_model_name(bqn_name, flt)
        pil_bqn = compute_pi_length([
            bqn_storage.sample_ensemble('test', ensemble_size=ENSEMBLE_SIZE)
            for _ in range(NUM_SAMPLES)
        ], get_reference_proabilities(reference_ensemble_size))
        lines = _pil_lineplot(axs[i], pil_bqn,
                              plot_style_kws={'linestyle': '--', 'label': bqn_name, 'color': lines[0].get_color(),
                                              'alpha': 0.8})


def _plot_pil_difference_for_ensemble_models(axs, flt, i, reference_ensemble_size=20):
    drn_names = ['ED-DRN', 'ST-DRN']
    bqn_names = ['ED-BQN', 'ST-BQN']
    axs[i].grid(alpha=0.5)
    axs[i].axvline((reference_ensemble_size - 1) / (reference_ensemble_size + 1), linestyle='--', color='k', alpha=0.5, zorder=0)
    axs[i].set(title=f'{flt}h', yscale='linear', xlim=XLIM)
    reference_probabilities = get_reference_proabilities(reference_ensemble_size)
    benchmarks = _get_pil_for_reference(flt, reference_probabilities)
    for j, (drn_name, bqn_name) in enumerate(zip(drn_names, bqn_names)):
        drn_storage = LegacyStorage.from_model_name(drn_name, flt)
        pil_drn = compute_pi_length([
            drn_storage.sample_ensemble('test', ensemble_size=ENSEMBLE_SIZE)
            for _ in range(NUM_SAMPLES)
        ], reference_probabilities)
        lines = _pil_difference_lineplot(
            axs[i], pil_drn, benchmarks['DRN'],
            plot_style_kws={'linestyle': '-', 'label': drn_name, 'alpha': 0.8}
        )
        bqn_storage = LegacyStorage.from_model_name(bqn_name, flt)
        pil_bqn = compute_pi_length([
            bqn_storage.sample_ensemble('test', ensemble_size=ENSEMBLE_SIZE)
            for _ in range(NUM_SAMPLES)
        ], reference_probabilities)
        lines = _pil_difference_lineplot(
            axs[i], pil_bqn, benchmarks['BQN'],
            plot_style_kws={'linestyle': '--', 'label': bqn_name, 'color': lines[0].get_color(), 'alpha': 0.8}
        )


def _plot_pil_for_minimal(axs, flt, i, reference_ensemble_size=20):
    benchmarks = _get_pil_for_reference(flt, get_reference_proabilities(reference_ensemble_size))
    lines = _pil_lineplot(
        axs[i], benchmarks['DRN'],
        plot_style_kws={'linestyle': '-', 'label': 'DRN', 'alpha': 0.8}
    )
    lines = _pil_lineplot(
        axs[i], benchmarks['BQN'],
        plot_style_kws={'linestyle': '--', 'label': 'BQN', 'color': lines[0].get_color(), 'alpha': 0.8}
    )


def _get_pil_for_reference(flt, reference_probabilities):
    file_name = f'{get_results_base_path()}/gusts/schulz_lerch_original/complete/step{flt}.csv'
    drn_storage = CSVReader(file_name, LogisticPrediction)
    pil_drn = compute_pi_length(
        [drn_storage.sample_ensemble(ensemble_size=ENSEMBLE_SIZE)],
        reference_probabilities
    )
    bqn_storage = CSVReader(file_name, BernsteinQuantilePrediction)
    pil_bqn = compute_pi_length(
        [bqn_storage.sample_ensemble(ensemble_size=ENSEMBLE_SIZE)],
        reference_probabilities
    )
    return {'DRN': pil_drn, 'BQN': pil_bqn}


if __name__ == '__main__':
    plot_pi_length_for_paper(save_figures=True)
    plot_pil_difference(save_figures=True)