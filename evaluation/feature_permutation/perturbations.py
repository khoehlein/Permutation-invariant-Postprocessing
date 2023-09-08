import math
from typing import Type, List, Dict

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch import Tensor
from torch.utils.data import TensorDataset


class IPerturbation(object):

    @property
    def name(self):
        raise NotImplementedError()

    def perturb_dataset(self, dataset: TensorDataset) -> TensorDataset:
        raise NotImplementedError()

    def __init__(self, num_channels: int, perturbed_channels: List[int]):
        super().__init__()
        self.num_channels = num_channels
        self.perturbed_channels= list(perturbed_channels)

    def perturbation_key(self):
        return get_perturbation_key(self.num_channels, self.perturbed_channels)


class ScalarPerturbation(IPerturbation):

    def get_scalar_predictor(self, dataset: Dict[str, np.ndarray], channel: int):
        assert dataset['dir_input'].shape[-1] == self.num_channels
        return dataset['dir_input'][:, channel]

    def reset_scalar_predictor(self, dataset: Dict[str, np.ndarray], channel: int, new_data: np.ndarray):
        assert dataset['dir_input'].shape[-1] == self.num_channels
        dataset['dir_input'][:, channel] = new_data
        return dataset


class ScalarPredictorPermutation(ScalarPerturbation):

    def __init__(self, num_channels: int, perturbed_channels: List[int], seed: int = None):
        super().__init__(num_channels, perturbed_channels)
        self.generator = np.random.Generator(np.random.PCG64(seed=seed))

    @property
    def name(self):
        return f'scalar-shuffle_{self.perturbation_key()}'

    def perturb_dataset(self, dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for channel in self.perturbed_channels:
            old_data = self.get_scalar_predictor(dataset, channel)
            new_data = self.perturb_array(old_data)
            self.reset_scalar_predictor(dataset, channel, new_data)
        return dataset

    def perturb_array(self, old_data: Tensor) -> Tensor:
        assert len(old_data.shape) == 1
        new_order = np.argsort(self.generator.random(size=len(old_data)))
        return old_data[new_order]


class EnsemblePerturbation(IPerturbation):

    def get_ensemble_predictor(self, dataset: TensorDataset, channel: int):
        assert dataset.tensors[0].shape[-1] == self.num_channels
        return dataset.tensors[0][..., channel]

    def reset_ensemble_predictor(self, dataset: TensorDataset, channel: int, new_data: Tensor):
        assert dataset.tensors[0].shape[-1] == self.num_channels
        dataset.tensors[0][..., channel] = new_data
        return dataset


def get_perturbation_key(num_channels, perturbed_channels):
    return ''.join([str(int(c in perturbed_channels)) for c in range(num_channels)])


class EnsembleRankShuffle(EnsemblePerturbation):

    def __init__(self, num_channels: int, perturbed_channels: List[int], seed: int = None):
        super().__init__(num_channels, perturbed_channels)
        self.generator = np.random.Generator(np.random.PCG64(seed=seed))

    @property
    def name(self):
        return f'ensemble-rank-shuffle_{self.perturbation_key()}'

    def perturb_dataset(self, dataset: TensorDataset) -> TensorDataset:
        for channel in self.perturbed_channels:
            old_data = self.get_ensemble_predictor(dataset, channel)
            new_data = self.perturb_tensor(old_data)
            self.reset_ensemble_predictor(dataset, channel, new_data)
        return dataset

    def perturb_tensor(self, old_data: Tensor) -> Tensor:
        assert len(old_data.shape) == 2
        num_samples, ensemble_size = old_data.shape
        new_order = np.argsort(self.generator.random(size=old_data.shape), axis=-1)
        sample_index = np.arange(num_samples)[:, None]
        return old_data[sample_index, new_order]


class Ranking(object):

    def __init__(self, data: np.ndarray, method='unique'):
        super().__init__()
        self.data = data
        self.method = method
        self.ranks = self._get_ranks(data)

    def get_bins(self, num_bins: int):
        return np.floor(self.ranks * num_bins).astype(int)

    def _get_ranks(self, data: np.ndarray):
        if self.method == 'unique':
            values, uranks = np.unique(data, return_inverse=True)
            ranks = (uranks + 1) / (len(values) + 1)
        else:
            ranks = rankdata(data, method=self.method) / (len(data) + 1)
        return ranks


class BinnedShuffle(EnsemblePerturbation):

    def __init__(self, num_channels: int, perturbed_channels: List[int], seed: int = None, preserve_ranking: bool = True):
        super().__init__(num_channels, perturbed_channels)
        self.generator = np.random.Generator(np.random.PCG64(seed=seed))
        self.preserve_ranking = preserve_ranking

    def _compute_binning(self, data: np.ndarray, num_bins: int) -> np.ndarray:
        if num_bins is None:
            return np.zeros(data.shape, dtype=int)
        return Ranking(data, method='unique').get_bins(num_bins)

    def _shuffle(self, old_data: Tensor, binning):
        num_samples = len(old_data)
        new_index = np.arange(num_samples)
        for bin in binning.keys():
            samples_in_bin = binning.get(bin)
            permutation = self.generator.permutation(samples_in_bin)
            new_index[samples_in_bin] = permutation
        new_data = old_data[new_index]
        if self.preserve_ranking:
            ranks = torch.argsort(torch.argsort(old_data, dim=-1), dim=-1)
            new_data = torch.sort(new_data, dim=-1).values
        else:
            ranks = torch.argsort(torch.from_numpy(self.generator.random(size=new_data.shape)), dim=-1)
        sample_index = torch.arange(num_samples, dtype=int).unsqueeze(-1)
        return new_data[sample_index, ranks]


class MetricPreservationShuffle(BinnedShuffle):

    METRICS = {
        'mean': lambda x: torch.mean(x, dim=-1),
        'median': lambda x: torch.median(x, dim=-1).values,
        'min': lambda x: torch.min(x, dim=-1).values,
        'max': lambda x: torch.max(x, dim=-1).values,
        'std': lambda x: torch.std(x, dim=-1, unbiased=True),
        'iqr': lambda x: torch.quantile(x, 0.75, dim=-1) - torch.quantile(x, 0.25, dim=-1),
        'range': lambda x: torch.max(x, dim=-1).values - torch.min(x, dim=-1).values,
        'skew': lambda x: torch.mean(torch.pow(x - torch.mean(x, dim=-1, keepdim=True), 3.), dim=-1) / torch.pow(torch.var(x, dim=-1, unbiased=True), 1.5),
        'kurt': lambda x: torch.mean(torch.pow(x - torch.mean(x, dim=-1, keepdim=True), 4.), dim=-1) / torch.var(x, dim=-1, unbiased=True) ** 2.,
    }

    @classmethod
    def available_metrics(cls):
        return list(cls.METRICS.keys())

    def __init__(
            self,
            num_channels: int, perturbed_channels: List[int],
            num_bins: int = None, metric: str = 'mean',
            preserve_ranking: bool = True, seed: int = None
    ):
        super().__init__(
            num_channels, perturbed_channels,
            seed=seed, preserve_ranking=preserve_ranking
        )
        self.num_bins = num_bins
        self.metric = metric.lower()
        assert self.metric in self.METRICS

    @property
    def name(self):
        statement = f'{self.metric}-{self.num_bins}'
        filler = '_ranked_' if self.preserve_ranking else '_'
        return f'ensemble-shuffle_{statement}{filler}{self.perturbation_key()}'

    def perturb_dataset(self, dataset: TensorDataset) -> TensorDataset:
        for channel in self.perturbed_channels:
            old_data = self.get_ensemble_predictor(dataset, channel)
            new_data = self.perturb_tensor(old_data)
            self.reset_ensemble_predictor(dataset, channel, new_data)
        return dataset

    def perturb_tensor(self, old_data: Tensor):
        with torch.no_grad():
            metrics = self.METRICS[self.metric](old_data).data.cpu().numpy()
            bins = self._compute_binning(metrics, self.num_bins)
            binning = pd.DataFrame({'bin': bins, 'metric': metrics,}).groupby(by='bin').groups
        new_data = self._shuffle(old_data, binning)
        return new_data


class LocationScaleShuffle(BinnedShuffle):

    LOC_METRICS = {
        'mean': lambda x: torch.mean(x, dim=-1),
        'median': lambda x: torch.median(x, dim=-1),
        'min': lambda x: torch.min(x, dim=-1),
        'max': lambda x: torch.max(x, dim=-1),
    }

    SCALE_METRICS ={
        'std': lambda x: torch.std(x, dim=-1, unbiased=True),
        'iqr': lambda x: torch.quantile(x, 0.75, dim=-1) - torch.quantile(x, 0.25, dim=-1),
        'range': lambda x: torch.max(x, dim=-1) - torch.min(x, dim=-1),
    }

    def __init__(
            self,
            num_channels: int, perturbed_channels: List[int],
            num_loc_bins: int = None, num_scale_bins: int = None,
            loc_metric: str = 'mean', scale_metric: str = 'std',
            preserve_ranking: bool = True, seed: int = None
    ):
        super().__init__(
            num_channels, perturbed_channels,
            seed=seed, preserve_ranking=preserve_ranking
        )
        self.num_loc_bins = num_loc_bins
        self.num_scale_bins = num_scale_bins
        self.loc_metric = loc_metric.lower()
        assert self.loc_metric in self.LOC_METRICS
        self.scale_metric = scale_metric.lower()
        assert self.scale_metric in self.SCALE_METRICS

    @property
    def name(self):
        loc_statement = f'loc-{self.loc_metric}-{self.num_loc_bins}'
        scale_statement = f'scale-{self.scale_metric}-{self.num_scale_bins}'
        filler = '_ranked_' if self.preserve_ranking else '_'
        return f'ensemble-shuffle_{loc_statement}_{scale_statement}{filler}{self.perturbation_key()}'

    def perturb_dataset(self, dataset: TensorDataset) -> TensorDataset:
        for channel in self.perturbed_channels:
            old_data = self.get_ensemble_predictor(dataset, channel)
            new_data = self.perturb_tensor(old_data)
            self.reset_ensemble_predictor(dataset, channel, new_data)
        return dataset

    def perturb_tensor(self, old_data: Tensor):
        with torch.no_grad():
            loc = self.LOC_METRICS[self.loc_metric](old_data).data.cpu().numpy()
            loc_bins = self._compute_binning(loc, self.num_loc_bins)
            scale = self.SCALE_METRICS[self.scale_metric](old_data).data.cpu().numpy()
            scale_bins = self._compute_binning(scale, self.num_scale_bins)
        binning = pd.DataFrame({'loc_bin': loc_bins,'scale_bin': scale_bins,}).groupby(by=['loc_bin','scale_bin']).groups
        new_data = self._shuffle(old_data, binning)
        return new_data

    @staticmethod
    def _get_fraction_bins(data: np.ndarray, num_bins: int) -> np.ndarray:
        pit = rankdata(data) / (len(data) + 1)
        bins = np.floor(pit * num_bins).astype(int)
        return bins

    @staticmethod
    def _get_sample_bins(data: np.ndarray, samples_per_bin: int) -> np.ndarray:
        bins = np.zeros(data.shape, dtype=int)
        order = np.argsort(data)
        num_bins = math.ceil(len(data) / samples_per_bin)
        for i, section in enumerate(np.array_split(order, num_bins)):
            bins[section] = i
        return bins


def _test():
    perturbation = LocationScaleShuffle(6, [0, 1 , 2], num_loc_bins=10)
    sample_data = torch.randn(100, 4, 6)
    perturbation.perturb_tensor(sample_data[...,0])
    print('Done')


class PerturbationFactory(object):

    def __init__(self, perturbation_class: Type, **kwargs):
        self.perturbation_class= perturbation_class
        self.kwargs = kwargs

    def generate(self, num_channels: int, perturbed_channels: List[int]):
        return self.perturbation_class(num_channels, perturbed_channels, **self.kwargs)
