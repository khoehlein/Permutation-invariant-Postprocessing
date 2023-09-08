import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

from experiments.baselines.bqn_utils import BernsteinQuantilePrediction
from experiments.baselines.drn_utils import LogisticPrediction, NormalPrediction
from utils.automation.storage import MultiRunExperiment

SEED = 42
ENSEMBLE_SIZE = 10
NUM_SAMPLES = 50

gen = np.random.Generator(np.random.PCG64(SEED))


def _cast_to_prediction(y_pred: np.ndarray, mode: str, num_samples=11):
    prediction = {
        'logistic': LogisticPrediction,
        'normal': NormalPrediction,
        'bqn': BernsteinQuantilePrediction
    }[mode](y_pred)
    if isinstance(prediction, BernsteinQuantilePrediction):
        prediction = prediction.to_ensemble(num_samples)
    return prediction


def load_predictions(path: str):
    files = [f for f in os.listdir(path) if f.endswith('.npz')]
    return [np.load(os.path.join(path,file)) for file in sorted(files)]


def sample_ensemble(predictions: list, mode: str, num_members = ENSEMBLE_SIZE):
    choice = gen.choice(np.arange(len(predictions)), num_members)
    predictions = [predictions[c]['predictions'] for c in choice]
    y_pred = sum(predictions) / len(predictions)
    return _cast_to_prediction(y_pred, mode)


def compute_coverage(predictions: list, observations: np.ndarray, cover_probs: np.ndarray):
    data = np.zeros((len(predictions), len(cover_probs)))
    with tqdm.tqdm(total=int(np.prod(data.shape))) as pbar:
        for i, y_pred in enumerate(predictions):
            for j, p in enumerate(cover_probs):
                data[i, j] = y_pred.compute_coverage(observations, alpha=(1.-p))
                pbar.update(1)
    return data


def _plot_coverage(ax, coverage: np.ndarray, coverage_probs: np.ndarray, residual: bool):
    assert coverage.shape[-1] == len(coverage_probs)
    if residual:
        coverage = coverage - coverage_probs
    ax.boxplot(coverage, positions=coverage_probs, widths=np.min(np.abs(np.diff(coverage_probs)))*0.8)
    if residual:
        ax.axhline(0., linestyle='--', color='k')
    else:
        mima = [0., 1.]
        ax.plot(mima, mima, linestyle='--', color='k')
    ylabel = 'Coverage error' if residual else 'Coverage (observed)'
    xticks = np.round(coverage_probs, decimals=2)
    ax.set(
        xlabel='Coverage (theory)', ylabel=ylabel,
        xlim=[-0.05,1.05],
        xticks=xticks
    )
    ax.set_xticklabels(xticks, rotation=45)


def draw_coverage_plot(ax, predictions: list, observations: np.ndarray, coverage_probs: np.array, residual=False):
    coverage = compute_coverage(predictions, observations, coverage_probs)
    _plot_coverage(ax, coverage, coverage_probs, residual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',type=str, choices=['logistic', 'normal', 'bqn'])
    parser.add_argument('--data-fold', type=str, default='test', choices=['test', 'valid', 'forecasts'])
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--num-members', type=int, required=True)
    args = vars(parser.parse_args())
    path = args['path']
    mode = args['mode']
    data_fold = args['data_fold']

    print(f'[INFO] Evaluating {path}')
    experiment = MultiRunExperiment(path)
    params = pd.DataFrame(experiment.list_run_parameters())
    params = params.drop([
        c for c in params.columns
        if
        (c not in {'run_name', 'time_stamp'}) and (np.all(params[c] == params[c].iloc[0]) or np.all(params[c].isnull()))
    ], axis=1)
    cover_probs = np.arange(args['num_members'] - 1, 1, -2) / (args['num_members'] + 1)
    for i in range(len(params)):
        print('[INFO] Parameters:')
        print(params.iloc[i])
        run = experiment.load_run(params['run_name'].iloc[i])
        npz_path = os.path.join(run.get_evaluation_path(), 'predictions', data_fold)
        predictions = load_predictions(npz_path)
        observations = predictions[0]['obs']
        predictions = [
            sample_ensemble(predictions, mode)
            for _ in range(NUM_SAMPLES)
        ]
        fig, ax = plt.subplots(1, 1, dpi=300, figsize=(8, 4))
        fig.suptitle(f'{path} ({data_fold})')
        draw_coverage_plot(ax, predictions, observations, cover_probs, residual=True)
        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()
