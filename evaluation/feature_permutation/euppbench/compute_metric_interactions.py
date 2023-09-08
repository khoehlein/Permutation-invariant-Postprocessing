import os

import numpy as np
import xarray as xr
import torch
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import FastICA

from data.config.interface import get_results_base_path
from evaluation.feature_permutation.perturbations import MetricPreservationShuffle
from data.euppbench.reforecasts import (
    DEFAULT_SPLIT as EUPP_SPLIT, DataConfig as EUPPConfig, build_training_dataset as build_eupp_dataset
)
from data.cosmo_de import (
    DEFAULT_SPLIT as GUSTS_SPLIT, DataConfig as GustsConfig, build_training_dataset as build_gusts_dataset
)


def build_sample_data(dataset: str, flt: int):
    device = torch.device('cpu')
    if dataset == 'eupp':
        args = {
            'data:flt': flt,
            'data:target': 't2m',
            'data:predictors': None,
            'data:splitting': EUPP_SPLIT,
            'data:target_completeness': 0.5,
            'data:location_embedding': 'spherical'
        }
        channel_labels = EUPPConfig.from_args(args)._predictor_names()
        data, conditions = build_eupp_dataset(args, device, test=True)
    elif dataset =='gusts':
        args = {
            'data:flt': flt,
            'data:target': 'wind_speed_of_gust',
            'data:predictors': None,
            'data:splitting': GUSTS_SPLIT,
            'data:location_embedding': 'spherical'
        }
        channel_labels = GustsConfig.from_args(args)._predictor_names()
        data, conditions = build_gusts_dataset(args, device, test=True)
    else:
        raise ValueError(f'[ERROR] Unknown dataset: {dataset}')
    data = data[-1]
    return data, channel_labels


def compute_ica(dataset: str, flt: int):
    output_path = f'{get_results_base_path()}/wge/{dataset}/cache/perturbations'
    os.makedirs(output_path, exist_ok=True)
    data, channel_labels = build_sample_data(dataset, flt)
    num_channels = len(channel_labels)
    metrics = [m for m in MetricPreservationShuffle.available_metrics() if m != 'range']
    with torch.no_grad():
        print('[INFO] Computing reference.')
        reference = {
            metric_name: torch.stack([
                MetricPreservationShuffle.METRICS[metric_name](data.tensors[0][..., i])
                for i in range(num_channels)
            ], dim=-1).data.cpu().numpy()
            for metric_name in metrics
        }

        for i, channel_name in enumerate(channel_labels):
            metric_data = np.stack([
                reference[metric_pert][..., i]
                for j, metric_pert in enumerate(metrics)
            ], axis=-1)
            metric_data = metric_data[~np.any(np.logical_or(np.isnan(metric_data), np.isinf(metric_data)), axis=-1)]
            ica = FastICA(whiten='unit-variance')
            ica.fit(metric_data)
            plt.figure()
            plt.suptitle(channel_name)
            plt.plot(ica.mixing_)
            plt.gca().set(xticks=np.arange(ica.n_features_in_))
            plt.gca().set_xticklabels(metrics, rotation=90)
            plt.show()
            plt.close()


def precompute_interaction_metrics(dataset: str, flt: int):
    output_path = f'{get_results_base_path()}/{dataset}/cache/perturbations'
    os.makedirs(output_path, exist_ok=True)
    data, channel_labels = build_sample_data(dataset, flt)
    num_channels = len(channel_labels)
    metrics = list(MetricPreservationShuffle.available_metrics())
    num_metrics = len(metrics)
    num_bins = 100
    with torch.no_grad():
        print('[INFO] Computing reference.')
        reference = {
            metric_name: torch.stack([
                MetricPreservationShuffle.METRICS[metric_name](data.tensors[0][..., i])
                for i in range(num_channels)
            ], dim=-1).data.cpu().numpy()
            for metric_name in metrics
        }
        correlations = np.zeros((num_channels, num_metrics, num_metrics))
        pvalues = np.zeros((num_channels, num_metrics, num_metrics))
        for i, channel_name in enumerate(channel_labels):
            for j, metric_pert in enumerate(metrics):
                print(f'[INFO] Computing perturbation: {metric_pert}')
                shuffler = MetricPreservationShuffle(
                    num_channels, perturbed_channels=[i],
                    num_bins=num_bins, metric=metric_pert, preserve_ranking=True,
                    seed=42
                )
                old_data = shuffler.get_ensemble_predictor(data, i)
                new_data = shuffler.perturb_tensor(old_data)
                for k, metric_ref in enumerate(metrics):
                    perturbed_metric = shuffler.METRICS[metric_ref](new_data).data.cpu().numpy()
                    reference_metric = reference[metric_ref][..., i]
                    c, p = spearmanr(perturbed_metric, reference_metric, nan_policy='omit')
                    correlations[i, j, k], pvalues[i, j, k] = c, p
        print('[INFO] Building dataset')
        ds = xr.Dataset(
            data_vars={
                'spearman': (['channel', 'perturbation', 'reference'], correlations),
                'pvalue': (['channel', 'perturbation', 'reference'], pvalues),
            },
            coords={
                'perturbation': (['perturbation'], metrics),
                'reference': (['reference'], metrics),
                'channel': (['channel'], channel_labels),
            },
            attrs={'description': f'Bin size {num_bins}'}
        )
        ds.to_netcdf(os.path.join(output_path, f'metric_interactions_{flt}h.nc'))


if __name__ == '__main__':
    dataset = 'gusts'
    lead_times = [6, 12, 18]
    for flt in lead_times:
        precompute_interaction_metrics(dataset, flt)
