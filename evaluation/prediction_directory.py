import os
from typing import Type

import numpy as np
import pandas as pd
import xarray as xr

from evaluation.config.interface import get_paths
from experiments.baselines.bqn_utils import BernsteinQuantilePrediction
from experiments.baselines.drn_utils import LogisticPrediction, NormalPrediction
from utils.automation.storage import MultiRunExperiment, PyTorchRun


SEED = 42
GENERATOR = np.random.Generator(np.random.PCG64(SEED))
SELECTED_RUNS_EUPP = get_paths('euppbench')
SELECTED_RUNS_GUSTS = get_paths('cosmo_de')
SELECTED_RUNS_GUSTS_LEGACY = get_paths('cosmo_de_legacy')
PREDICTION_TYPE = {
    'logistic': LogisticPrediction,
    'normal': NormalPrediction,
    'bqn': BernsteinQuantilePrediction
}


def prediction_type_from_model_name(model_name: str):
    if 'BQN' in model_name:
        return 'bqn'
    if 'DRN' in model_name:
        if 'TN' in model_name:
            return 'normal'
        return 'logistic'
    raise ValueError(f'[ERROR] Unknown model key: {model_name}')


class PredictionStorage(object):

    @classmethod
    def from_model_name(cls, model_name: str, flt: int, perturbation: str = None, dataset=None):
        if dataset is None:
            dataset = 'eupp'
        selected_runs_dict = {
            'eupp': SELECTED_RUNS_EUPP,
            'gusts': SELECTED_RUNS_GUSTS,
        }[dataset]
        assert model_name in selected_runs_dict.keys()
        prediction_type_name = prediction_type_from_model_name(model_name)
        return cls.from_run_path(selected_runs_dict[model_name][flt], prediction_type_name, perturbation=perturbation)

    @classmethod
    def from_run_path(cls, run_path: str, prediction_type_name: str, perturbation: str =None):
        experiment, run = MultiRunExperiment.from_run_path(run_path, _except_on_not_existing=True, return_run=True)
        return cls(run, PREDICTION_TYPE[prediction_type_name], perturbation=perturbation)

    def __init__(self, run: PyTorchRun, prediction_class: Type, perturbation: str = None):
        self.run = run
        self.prediction_class = prediction_class
        self._model_predictions = {dataset: None for dataset in ['test', 'valid', 'forecasts']}
        self.perturbation = perturbation

    def get_cache_dir(self):
        path = os.path.join(self.run.get_evaluation_path(), 'cache')
        os.makedirs(path, exist_ok=True)
        return path

    def num_models(self, dataset: str):
        return len(self.get_model_predictions(dataset))

    def get_model_predictions(self, dataset: str):
        assert dataset in self._model_predictions.keys()
        if self._model_predictions[dataset] is None:
            prediction_path = self._get_path_to_predictions(dataset)
            files = [f for f in os.listdir(prediction_path) if f.endswith('.npz')]
            self._model_predictions[dataset] = [np.load(os.path.join(prediction_path, file)) for file in sorted(files)]
        return self._model_predictions[dataset]

    def _get_path_to_predictions(self, dataset: str):
        rel_path = ['predictions']
        if self.perturbation is not None:
            rel_path += ['perturbed', self.perturbation]
        return os.path.join(self.run.get_evaluation_path(), *rel_path, dataset)

    def get_observations(self, dataset: str):
        self.get_model_predictions(dataset)
        return self._model_predictions[dataset][0]['obs']

    def sample_ensemble(self, dataset: str, ensemble_size: int = 10, generator=None):
        if generator is None:
            generator = GENERATOR
        predictions = self.get_model_predictions(dataset)
        choice = generator.choice(np.arange(len(predictions)), ensemble_size, replace=False)
        prediction_data = [self._get_prediction_data(predictions[c]) for c in choice]
        y_pred = sum(prediction_data) / len(prediction_data)
        return self.prediction_class(y_pred)

    def _get_prediction_data(self, prediction):
        return prediction['predictions']

    def get_prediction(self,dataset: str, i: int):
        predictions = self.get_model_predictions(dataset)
        prediction_data = self._get_prediction_data(predictions[i])
        return self.prediction_class(prediction_data)


class LegacyStorage(object):

    ALIASSES = {
        'test': 'test',
        'valid': 'validation'
    }

    @classmethod
    def from_model_name(cls, model_name: str, flt: int):
        assert model_name in SELECTED_RUNS_GUSTS_LEGACY.keys()
        prediction_type_name = prediction_type_from_model_name(model_name)
        prediction_type = PREDICTION_TYPE[prediction_type_name]
        if 'bqn' in model_name.lower():
            filler = 'alphas'
            ext = '.nc'
        elif 'drn' in model_name.lower():
            filler = 'singleton'
            ext = '.csv'
        else:
            raise ValueError(f'[ERROR] Unsupported model name: {model_name}')
        file_name_pattern = '{basename}_' + filler + '_{dataset}' + ext
        return cls(SELECTED_RUNS_GUSTS_LEGACY[model_name][flt], prediction_type, file_name_pattern)

    def __init__(self, path: str, prediction_class: Type, file_name_pattern: str):
        self.path = os.path.abspath(path)
        self.basename = os.path.basename(self.path)
        self.prediction_class = prediction_class
        self.file_name_pattern = file_name_pattern
        self._model_predictions = {key: None for key in ['test', 'valid']}

    def get_model_predictions(self, dataset: str):
        if self._model_predictions[dataset] is None:
            file_name = self.file_name_pattern.format(basename=self.basename, dataset=self.ALIASSES[dataset])
            reader = self._get_file_reader(os.path.join(self.path, file_name))
            self._model_predictions[dataset] = reader
        return self._model_predictions[dataset]

    def _get_file_reader(self, file_path: str):
        _, ext = os.path.splitext(file_path)
        if ext == '.csv':
            return CSVReader(file_path, self.prediction_class)
        elif ext == '.nc':
            return NetCDFReader(file_path, self.prediction_class)
        raise NotImplementedError(f'[ERROR] Unknown extension: {ext}')

    def get_observations(self, dataset: str):
        return self.get_model_predictions(dataset).get_observations()

    def num_models(self, dataset: str):
        return self.get_model_predictions(dataset).num_models()

    def sample_ensemble(self, dataset: str, ensemble_size: int = 10, generator=None):
        reader = self.get_model_predictions(dataset)
        return reader.sample_ensemble(ensemble_size=ensemble_size, generator=generator)


class CSVReader(object):

    def __init__(self, csv_path: str, prediction_class: Type):
        self.csv_path = csv_path
        self.prediction_class = prediction_class
        self._obs = None
        self._model_predictions = None

    def get_model_predictions(self):
        if self._model_predictions is None:
            head = pd.read_csv(self.csv_path, nrows=2)
            index_col = None if 'obs' in head.columns else 0
            data = pd.read_csv(self.csv_path, index_col=index_col).rename(columns={'target': 'obs'})
            self._obs = data['obs'].values
            if 'drn_loc' in head.columns or 'bqn_alpha0' in head.columns:
                self._read_single_model_data(data)
            else:
                self._read_multi_model_data(data)
        return self._model_predictions

    def get_observations(self, data_fold=None):
        if data_fold is not None:
            assert data_fold == 'test', '[ERROR] CSVReader supports test data only'
        self.get_model_predictions()
        return self._obs

    def num_models(self):
        return len(self.get_model_predictions())

    def _read_single_model_data(self, data: pd.DataFrame):
        if self.prediction_class in {LogisticPrediction, NormalPrediction}:
            columns = ['drn_loc', 'drn_scale']
        elif self.prediction_class == BernsteinQuantilePrediction:
            alpha_cols = [c for c in data.columns if c.startswith('bqn_alpha')]
            num_alphas = len(alpha_cols)
            columns = [f'bqn_alpha{i}' for i in range(num_alphas)]
        else:
            raise NotImplementedError()
        self._model_predictions = [data.loc[:, columns].values]

    def _read_multi_model_data(self, data: pd.DataFrame):
        assert self.prediction_class == LogisticPrediction
        sigma_cols = [c for c in data.columns if c.startswith('sigma_')]
        num_models = len(sigma_cols)
        self._model_predictions = [
            data.loc[:, ['mu_{:03d}'.format(i), 'sigma_{:03d}'.format(i)]].values
            for i in range(num_models)
        ]

    def sample_ensemble(self, ensemble_size=10, generator=None):
        if generator is None:
            generator = GENERATOR
        self.get_model_predictions()
        num_models = len(self._model_predictions)
        if num_models == 1:
            # handle single-model case
            y_pred = self._model_predictions[0]
        elif num_models < ensemble_size:
            raise RuntimeError(f'[ERROR] Requested ensemble of size {ensemble_size}, but only {num_models} models are available')
        else:
            choice = generator.choice(np.arange(num_models), ensemble_size, replace=False)
            predictions = [self._model_predictions[int(c)] for c in choice]
            y_pred = sum(predictions) / len(predictions)
        return self.prediction_class(y_pred)


class NetCDFReader(object):

    def __init__(self, nc_path: str, prediction_class: Type):
        self.nc_path = nc_path
        self.prediction_class = prediction_class
        self._obs = None
        self._model_predictions = None

    def get_model_predictions(self):
        if self._model_predictions is None:
            if self.prediction_class != BernsteinQuantilePrediction:
                raise NotImplementedError()
            data = xr.open_dataset(self.nc_path)
            self._obs = data['target'].values
            predictions = np.moveaxis(data['alphas'].values, -1, 0)
            self._model_predictions = list(predictions)
        return self._model_predictions

    def get_observations(self, data_fold=None):
        if data_fold is not None:
            assert data_fold == 'test', '[ERROR] CSVReader supports test data only'
        self.get_model_predictions()
        return self._obs

    def num_models(self):
        return len(self.get_model_predictions())

    def sample_ensemble(self, ensemble_size=10, generator=None):
        if generator is None:
            generator = GENERATOR
        self.get_model_predictions()
        num_models = len(self._model_predictions)
        if num_models < ensemble_size:
            raise RuntimeError(f'[ERROR] Requested ensemble of size {ensemble_size}, but only {num_models} models are available')
        choice = generator.choice(np.arange(num_models), ensemble_size, replace=False)
        predictions = [self._model_predictions[int(c)] for c in choice]
        y_pred = sum(predictions) / len(predictions)
        return self.prediction_class(y_pred)
