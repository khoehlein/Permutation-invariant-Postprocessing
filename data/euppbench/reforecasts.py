import argparse
import dataclasses
import json
import os
import shutil
import uuid
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import TensorDataset

from data.config.interface import get_path

DEFAULT_SPLIT = '12:4:4'
DATASET_ROOT = get_path('euppbench_reforecasts_data')
TRAINING_CACHE_DIR = get_path('euppbench_reforecasts_cache')

DYNAMIC_PREDICTORS = [
    'cape', 'cp6',
    'mn2t6', 'mx2t6',
    'p10fg6',
    'sd',
    'slhf6', 'sshf6',
    'ssr6', 'ssrd6',
    'stl1', 'str6', 'strd6',
    'swvl1',
    't2m', 't850',
    'tcc', 'tcw', 'tcwv',
    'tp6',
    'u10', 'u100', 'u700',
    'v10', 'v100', 'v700',
    'vis', 'z500'
]

STATIC_PREDICTORS = [
    'station_altitude',
    'elevation_difference',
    'station_int_id'
]

ENSEMBLE_SIZE = 11


class Dimension:
    PRESSURE_LEVEL = 'isobaricInhPa'
    STATION_ID = 'station_id'
    MEMBER_ID = 'number'
    FLT = 'step'
    TIME = 'time'
    YEAR = 'year'
    SAMPLE = 'sample'
    PREDICTOR = 'predictor'


class VariableGroup:
    PRESSURE_LEVELS ='pressure'
    SURFACE = 'surface'
    SURFACE_PROCESSED = 'surface-processed'
    OBSERVATIONS = 'observations'
    OBSERVATIONS_PROCESSED = 'observations-processed'


@dataclasses.dataclass(unsafe_hash=True)
class DataConfig(object):
    flt: int
    target: str
    predictors: Union[None, str]
    splitting: str  # -> train, val, test
    completeness: float
    location_embedding: str

    def to_dict(self):
        raw_dict = dataclasses.asdict(self)
        return raw_dict

    def _predictor_names(self):
        if self.predictors is None:
            return DYNAMIC_PREDICTORS
        return self.predictors.split(':')

    def num_channels(self):
        return len(self._predictor_names())

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data')
        group.add_argument('--data:flt', type=int, help="forcast lead time", required=True)
        group.add_argument('--data:predictors', type=str,
                           help='names of (dynamic) predictor variables,separated by ":"', default=None)
        group.add_argument('--data:target', type=str, help='names of target variable', default='t2m')
        group.add_argument('--data:splitting', type=str, help='number of years for training, validation and test, separated by ":"', default=DEFAULT_SPLIT)
        group.add_argument('--data:target-completeness', type=float, help='threshold for station-wise completeness of target observations', default=0.5)
        group.add_argument('--data:location-embedding', type=str, help='transform for lat-lon station coordinates', default='plain', choices=['spherical', 'plain'])

    @classmethod
    def from_args(cls, args):
        config_dict = {
            'flt': args['data:flt'],
            'target': args['data:target'],
            'predictors': args['data:predictors'],
            'splitting': args['data:splitting'],
            'completeness': args['data:target_completeness'],
            'location_embedding': args['data:location_embedding'],
        }
        return cls(**config_dict)


class DataCache(object):

    class Entry(object):

        CONFIG_KEY = 'config.json'
        DATA_KEY = 'data'

        def __init__(self, config: DataConfig, path: Optional[str] = None):
            self.config = config
            self.path = path

        def dump(self, root_path: str, name: Optional[str] = None):
            if self.path is not None:
                return self.path
            dump_path = os.path.join(root_path, name or str(uuid.uuid4()))
            os.makedirs(dump_path)
            self.path = dump_path
            config_path = os.path.join(dump_path, self.CONFIG_KEY)
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, sort_keys=True, indent=4)
            os.makedirs(self._data_path)
            return self

        @property
        def _data_path(self):
            data_path = os.path.join(self.path, self.DATA_KEY)
            return data_path

        @classmethod
        def from_path(cls, path: str):
            config_path = os.path.join(path, cls.CONFIG_KEY)
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = DataConfig(**config_dict)
            return cls(config, path=path)

        def delete(self):
            shutil.rmtree(self.path)

        def keys(self):
            return list(sorted(os.listdir(self._data_path)))

        def __getitem__(self, item):
            return self.load_data(item)

        def __len__(self):
            return len(self.keys())

        def save_data(self, key: str, **data: Tensor):
            assert self.path is not None
            key_path = os.path.join(self._data_path, key)
            os.makedirs(key_path)
            for data_key, value in data.items():
                file_name = os.path.join(key_path, f'{data_key}.pt')
                torch.save(data[data_key], file_name)
            return key_path

        def load_data(self, key: str):
            key_path = os.path.join(self._data_path, key)
            assert os.path.isdir(key_path)
            contents = [f for f in os.listdir(key_path) if f.endswith('.pt')]
            data = {
                os.path.splitext(data_key)[0]: torch.load(os.path.join(key_path, data_key))
                for data_key in contents
            }
            return data

        def load_all(self):
            data = {key: self.load_data(key) for key in self.keys()}
            return data

    def __init__(self, root_path: str):
        root_path = os.path.abspath(root_path)
        self.root_path = root_path

    def clear(self, check=True):
        if check:
            response = input(f'[INFO] Program trying to clear cache at {self.root_path}. Proceed? (Y/N)\n')
        else:
            response = 'y'
        if response.lower() == 'y':
            shutil.rmtree(self.root_path)
            os.mkdir(self.root_path)

    def list_entries(self):
        contents = os.listdir(self.root_path)
        entries = [
            self.Entry.from_path(os.path.join(self.root_path, f))
            for f in contents
            if os.path.isdir(os.path.join(self.root_path, f))
        ]
        return entries

    def __contains__(self, item: DataConfig):
        configs = {entry.config for entry in self.list_entries()}
        return item in configs

    def get(self, item: DataConfig):
        entries = {entry.config: entry for entry in self.list_entries()}
        return entries.get(item, None)

    def create_entry(self, config: DataConfig, check_exists=True, overwrite=False):
        if check_exists or overwrite:
            entry = self.get(config)
            if check_exists:
                assert entry is None
            if overwrite and entry is not None:
                entry.delete()
        entry = self.Entry(config)
        entry.dump(self.root_path)
        return entry

    def load(self, config: DataConfig):
        entry = self.get(config)
        assert entry is not None
        return entry.load_all()


class ZarrLoader(object):

    @staticmethod
    def _load_directory_contents(path: str, drop_valid_time=True):
        contents = [os.path.join(path, c) for c in sorted(os.listdir(path)) if c.endswith('zarr')]
        data = xr.open_mfdataset(contents, concat_dim=Dimension.STATION_ID, combine='nested', engine='zarr')
        if drop_valid_time and 'valid_time' in data:
            data = data.drop_vars(['valid_time'])
        return data

    def _load_pl_data(self):
        pl_root = os.path.join(DATASET_ROOT, VariableGroup.PRESSURE_LEVELS)
        pressure_levels = sorted(os.listdir(pl_root))
        pl_data = []
        for level in pressure_levels:
            level_data = self._load_directory_contents(os.path.join(pl_root, level)).isel(**{Dimension.PRESSURE_LEVEL: 0})
            level_data = level_data.drop_vars(Dimension.PRESSURE_LEVEL).rename_vars({
                name: f'{name}{level}'
                for name in level_data.data_vars
            })
            pl_data.append(level_data)
        pl_data = xr.merge(pl_data)
        return pl_data

    def _load_surface_data(self, group: str, drop_valid_time=True):
        data = self._load_directory_contents(os.path.join(DATASET_ROOT, group), drop_valid_time=drop_valid_time)
        if 'surface' in data.dims:
            data = data.isel(surface=0).drop_vars(['surface'])
        if 'depthBelowLandLayer'in data.dims:
            data = data.isel(depthBelowLandLayer=0).drop_vars(['depthBelowLandLayer'])
        return data

    @staticmethod
    def _stack_data(data: xr.Dataset):
        return data.stack(**{Dimension.SAMPLE: [Dimension.YEAR, Dimension.TIME, Dimension.STATION_ID]})

    @staticmethod
    def _find_valid_stations(data: xr.DataArray, completeness: float):
        availability = (1. - data.isnull().astype(np.float32)).mean([Dimension.YEAR, Dimension.TIME])
        valid_stations = data[Dimension.STATION_ID].sel(**{Dimension.STATION_ID: availability > completeness})
        return valid_stations

    @staticmethod
    def _get_yday(valid_time: xr.DataArray):
        year_lower = valid_time.astype('datetime64[Y]')
        year_upper = (year_lower + np.timedelta64(1, 'Y')).astype('datetime64[h]')
        yday = (valid_time.astype('datetime64[h]') - year_lower) / (year_upper - year_lower.astype('datetime64[h]'))
        return yday.astype(np.float32)

    @staticmethod
    def _extract_static_data(data: xr.Dataset):
        lat = data['station_latitude']
        lon = data['station_longitude']
        alt = data['station_altitude']
        station_id = data['station_id']
        elv = data['station_altitude'] - data['model_orography']
        static_data = pd.DataFrame({
            'station_latitude': lat.values,
            'station_longitude': lon.values,
            'station_altitude': alt.values,
            'elevation_difference': elv.values,
            'station_id': station_id.values,
        })
        return static_data

    def load(self, config: DataConfig):
        print('[INFO] Loading data from disk.')
        targets = xr.merge([
            self._load_surface_data(VariableGroup.OBSERVATIONS),
            self._load_surface_data(VariableGroup.OBSERVATIONS_PROCESSED),
        ])[config.target].sel(**{Dimension.FLT: np.timedelta64(config.flt,'h')})
        valid_stations = self._find_valid_stations(targets, config.completeness)
        targets = targets.sel(**{Dimension.STATION_ID: valid_stations})
        predictors = xr.merge([
            self._load_pl_data(),
            self._load_surface_data(VariableGroup.SURFACE, drop_valid_time=False),
            self._load_surface_data(VariableGroup.SURFACE_PROCESSED),
        ]).sel(**{Dimension.FLT: np.timedelta64(config.flt,'h'), Dimension.STATION_ID: valid_stations})
        predictors['yday'] = self._get_yday(predictors['valid_time'])
        station_data = self._extract_static_data(predictors)
        predictors = self._stack_data(predictors)
        targets = self._stack_data(targets)
        valid = ~ np.logical_or(targets.isnull(), predictors['vis'].isnull().any(Dimension.MEMBER_ID))
        targets = targets.sel(**{Dimension.SAMPLE: valid})
        predictors = predictors.sel(**{Dimension.SAMPLE: valid})
        print('[INFO] Loading completed.')
        return predictors, targets, station_data

    def find_valid_stations(self, config: DataConfig):
        targets = xr.merge([
            self._load_surface_data(VariableGroup.OBSERVATIONS),
            self._load_surface_data(VariableGroup.OBSERVATIONS_PROCESSED),
        ])[config.target].sel(**{Dimension.FLT: np.timedelta64(config.flt, 'h')})
        valid_stations = self._find_valid_stations(targets, config.completeness)
        return valid_stations


class Preprocess(object):

    PREDICTOR_TARGET_MAPPING = {}
    DATA_KEYS = ['dynamic_ensemble', 'yday', 'station_index', 'static', 'targets']

    def __init__(self, config: DataConfig, scalers=None, predictor_target_mapping=None):
        self.config = config
        if scalers is None:
            scalers = {
                'static': StandardScaler(),
                'dynamic': StandardScaler(),
            }
        self.scalers = scalers

    @staticmethod
    def _split_once(data: xr.Dataset, split_idx: int):
        year = data.coords[Dimension.YEAR]
        lower = year <= split_idx
        return data.sel(**{Dimension.SAMPLE: lower}), data.sel(**{Dimension.SAMPLE: ~lower})

    def split(self, dataset: xr.Dataset):
        max_train, max_val, max_test = np.cumsum([int(s) for s in self.config.splitting.split(':')])
        data, _ = self._split_once(dataset, max_test)
        data_train, data_test = self._split_once(dataset, max_val)
        data_train, data_val = self._split_once(data_train, max_train)
        return {
            'train': data_train, 'val': data_val, 'test': data_test
        }

    def get_ensemble_predictors(self, predictors: xr.Dataset):
        predictor_names = self.config._predictor_names()
        data = [
            predictors[key].transpose(Dimension.SAMPLE, Dimension.MEMBER_ID).values
            for key in predictor_names
        ]
        return np.stack(data, axis=-1)

    def get_named_ensemble_predictors(self, predictors: xr.Dataset, mean=True):
        predictor_names = self.config._predictor_names()
        if mean:
            data = {
                str(key) + '_mean': predictors[key].mean(Dimension.MEMBER_ID).values.ravel()
                for key in predictor_names
            }
        else:
            data = {
                key: predictors[key].transpose(Dimension.SAMPLE, Dimension.MEMBER_ID).values
                for key in predictor_names
            }
        return data

    def get_station_predictors(self, data: pd.DataFrame):
        lat = data['station_latitude'].values
        lon = data['station_longitude'].values

        def _sperical_coords(lat, lon):
            sin_lat = np.sin(np.deg2rad(90. - lat))
            sin_lon = np.sin(np.deg2rad(lon))
            cos_lat = np.cos(np.deg2rad(90. - lat))
            cos_lon = np.cos(np.deg2rad(lon))
            xyz = np.empty((len(data), 3))
            xyz[:, 0] = sin_lat * sin_lon
            xyz[:, 1] = sin_lat * cos_lon
            xyz[:, 2] = cos_lat
            return xyz

        def _plain_lat_lon(lat, lon):
            return np.stack([lat.ravel(), lon.ravel()], axis=-1)

        station_data = np.empty((len(data), 4))
        station_data[:, 0] = data['station_altitude'].values
        station_data[:, 1] = data['elevation_difference'].values
        station_data[:, 2] = data['station_bias'].values
        station_data[:, 3] = data['station_coverage'].values

        pos_embedding = self.config.location_embedding
        if pos_embedding in {'spherical'}:
            lat_lon_data = _sperical_coords(lat, lon)
        elif pos_embedding in {'plain'}:
            lat_lon_data = _plain_lat_lon(lat, lon)
        else:
            raise Exception(f'[ERROR] Unknown embedding type: {pos_embedding}')

        return np.concatenate([station_data, lat_lon_data], axis=-1)

    def get_named_station_predictors(self, data: pd.DataFrame):
        lat = data['station_latitude'].values
        lon = data['station_longitude'].values

        def _sperical_coords(lat, lon):
            sin_lat = np.sin(np.deg2rad(90. - lat))
            sin_lon = np.sin(np.deg2rad(lon))
            cos_lat = np.cos(np.deg2rad(90. - lat))
            cos_lon = np.cos(np.deg2rad(lon))
            xyz = {
                'x': sin_lat * sin_lon,
                'y': sin_lat * cos_lon,
                'z': cos_lat,
            }
            return xyz

        def _plain_lat_lon(lat, lon):
            return {
                'lat': lat.ravel(),
                'lon': lon.ravel(),
            }

        station_data = {
            'alt': data['station_altitude'].values,
            'orog': data['elevation_difference'].values,
            'loc_bias': data['station_bias'].values,
            'loc_cover': data['station_coverage'].values
        }

        pos_embedding = self.config.location_embedding
        if pos_embedding in {'spherical'}:
            lat_lon_data = _sperical_coords(lat, lon)
        elif pos_embedding in {'plain'}:
            lat_lon_data = _plain_lat_lon(lat, lon)
        else:
            raise Exception(f'[ERROR] Unknown embedding type: {pos_embedding}')

        return {**station_data, **lat_lon_data}

    @staticmethod
    def _apply_to_ensemble(ensemble: np.ndarray, func):
        shape = ensemble.shape
        ensemble = ensemble.view()
        ensemble.shape = (-1, shape[-1])
        ensemble = func(ensemble).view()
        ensemble.shape = shape
        return np.asarray(ensemble)

    def get_station_index(self, predictors: xr.Dataset, station_data: pd.DataFrame):
        station_ids = predictors['station_id'].values
        df = pd.DataFrame({
            'station_id': station_data['station_id'],
            'station_index': station_data.index.values,
        }).set_index('station_id')
        station_index = df['station_index'].loc[station_ids]
        return station_index.values.astype(np.long)

    def compute_station_stats(self, predictors: xr.Dataset, targets: xr.DataArray, station_data: pd.DataFrame):
        print('[INFO] Computing station statistics.')
        prediction = self.get_primary_prediction(predictors, targets)
        prediction.load()
        targets.load()
        station_error = prediction.transpose(Dimension.SAMPLE, Dimension.MEMBER_ID).values - targets.values[:, None]
        num_members = station_error.shape[-1]
        p_cover = 1. - (2. / (num_members + 1))
        coverage = prediction.quantile(np.array([1. + p_cover, 1. - p_cover]) / 2., dim=Dimension.MEMBER_ID)
        is_covered = np.logical_and(coverage.isel(quantile=0).values >= targets.values, coverage.isel(quantile=1).values <= targets.values)
        station_stats = {
            'station_id': predictors['station_id'].values,
            'station_bias': np.mean(station_error, axis=-1),
            'station_coverage': is_covered.astype(np.float32),
        }
        station_stats = pd.DataFrame(station_stats).groupby('station_id').mean()
        station_data = pd.merge(station_data, station_stats, on='station_id')
        return station_data

    def get_primary_prediction(self, predictors: xr.Dataset, targets: xr.DataArray):
        predictor_name = self.PREDICTOR_TARGET_MAPPING.get(targets.name, targets.name)
        prediction = predictors[predictor_name]
        return prediction

    def transform(self, predictors: xr.Dataset, targets: xr.DataArray, station_data: pd.DataFrame, fit=False):
        ensemble = self.get_ensemble_predictors(predictors)
        static = self.get_station_predictors(station_data)
        if fit:
            print('[INFO] Fitting scalers.')
            ensemble = self._apply_to_ensemble(ensemble, self.scalers['dynamic'].fit_transform)
            static = self.scalers['static'].fit_transform(static)
            print('[INFO] Fitting completed.')
        else:
            ensemble = self._apply_to_ensemble(ensemble, self.scalers['dynamic'].transform)
            static = self.scalers['static'].transform(static)
        yday = predictors['yday'].values
        station_index = self.get_station_index(predictors, station_data)
        targets = self.select_target(targets)
        return {
            key: torch.from_numpy(value)
            for key, value in zip(
                self.DATA_KEYS,
                [ensemble, yday, station_index, static, targets]
            )
        }

    def to_data_frame(self, predictors: xr.Dataset, targets: xr.DataArray, station_data: pd.DataFrame):
        ensemble = self.get_named_ensemble_predictors(predictors, mean=True)
        prediction = self.get_primary_prediction(predictors, targets).transpose(Dimension.SAMPLE, Dimension.MEMBER_ID).values
        ens_mean = np.mean(prediction, axis=-1)
        ens_std = np.std(prediction, axis=-1, ddof=1)
        prediction = np.sort(prediction, axis=-1)
        prediction = {
            f'ens_{i + 1}': prediction[:, i]
            for i in range(prediction.shape[-1])
        }
        static = self.get_named_station_predictors(station_data)
        yday = np.cos(2. * np.pi * predictors['yday'].values)
        month = np.floor(12 * predictors['yday'].values) % 12
        station_index = self.get_station_index(predictors, station_data)
        targets = self.select_target(targets)
        data = {
            'obs': targets,
            'ens_mean': ens_mean,
            'ens_sd': ens_std,
            'yday': yday,
            'month': month,
            'location': station_index,
            **prediction,
            **ensemble,
            **{key: static[key][station_index] for key in static.keys()},
        }
        return pd.DataFrame(data)

    def select_target(self, targets: xr.DataArray):
        return targets.values


def _load_data_from_zarr(config: DataConfig, return_station_data=False):
    predictors, targets, station_data = ZarrLoader().load(config)
    predictors.load()
    targets.load()
    preprocess = Preprocess(config)
    data, targets = [preprocess.split(ds) for ds in [predictors, targets]]
    station_data = preprocess.compute_station_stats(data['train'], targets['train'], station_data)
    for key in ['train', 'val', 'test']:
        data[key] = preprocess.transform(data[key], targets[key], station_data, fit=(key == 'train'))
    if return_station_data:
        return data, preprocess.scalers, station_data
    return data, preprocess.scalers


def build_datasets(config: DataConfig, cache_dir=None, overwrite_cache=False, device=None):
    scalers = None
    if cache_dir is not None:
        cache = DataCache(cache_dir)
        if config in cache and not overwrite_cache:
            print('[INFO] Loading data from cache.')
            data = cache.load(config)
        else:
            data, scalers = _load_data_from_zarr(config)
            print('[INFO] Writing data to cache.')
            entry = cache.create_entry(config, overwrite=overwrite_cache)
            for key in data:
                entry.save_data(key, **data[key])
    else:
        data, scalers = _load_data_from_zarr(config)
    data, static = _to_torch_dataset(data, device)
    return data, static


def build_training_dataset(args, device, test=False):
    config = DataConfig.from_args(args)
    data, conditions = build_datasets(config, cache_dir=TRAINING_CACHE_DIR, overwrite_cache=False, device=device)
    if test:
        return (data['train'], data['val'], data['test']), conditions
    return (data['train'], data['val']), conditions


def _to_torch_dataset(data, device):
    datasets = {
        fold_key: TensorDataset(*[
            data[fold_key][data_key].to(device)
            for data_key in Preprocess.DATA_KEYS
            if data_key != 'static'
        ])
        for fold_key in data
    }
    static = data['train']['static'].to(device, dtype=torch.float32)
    return datasets, static


def _test():
    config = DataConfig(6, 't2m', None, '14:2:4', 0.5, 'spherical')
    data = _load_data_from_zarr(config)
    print('Done')
