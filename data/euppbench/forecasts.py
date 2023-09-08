import argparse
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.preprocessing import StandardScaler

from data.config.interface import get_path
from data.euppbench.reforecasts import (
    DataConfig,
    ZarrLoader as ReforecastLoader, DataCache,
    _load_data_from_zarr as _load_reforecasts, _to_torch_dataset
)


DATASET_ROOT = get_path('euppbench_forecasts_data')
TRAINING_CACHE_DIR = get_path('euppbench_forecasts_cache')
ENSEMBLE_SIZE = 51


class Dimension:
    PRESSURE_LEVEL = 'isobaricInhPa'
    STATION_ID = 'station_id'
    MEMBER_ID = 'number'
    FLT = 'step'
    TIME = 'time'
    SAMPLE = 'sample'
    PREDICTOR = 'predictor'


class VariableGroup:
    PRESSURE_LEVELS ='pressure'
    SURFACE = 'surface'
    SURFACE_PROCESSED = 'surface-processed'
    OBSERVATIONS = 'observations'
    OBSERVATIONS_PROCESSED = 'observations-processed'


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
        return data.stack(**{Dimension.SAMPLE: [Dimension.TIME, Dimension.STATION_ID]})

    @staticmethod
    def _find_valid_stations(config: DataConfig):
        valid_stations = ReforecastLoader().find_valid_stations(config)
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
        valid_stations = self._find_valid_stations(config)
        targets = targets.sel(**{Dimension.STATION_ID: valid_stations})
        predictors = xr.merge([
            self._load_pl_data(),
            self._load_surface_data(VariableGroup.SURFACE, drop_valid_time=False),
            self._load_surface_data(VariableGroup.SURFACE_PROCESSED),
        ]).sel(**{Dimension.FLT: np.timedelta64(config.flt,'h'), Dimension.STATION_ID: valid_stations})
        predictors['yday'] = self._get_yday(predictors['valid_time'])
        predictors = self._stack_data(predictors)
        targets = self._stack_data(targets)
        valid = ~ np.logical_or(targets.isnull(), predictors['vis'].isnull().any(Dimension.MEMBER_ID))
        targets = targets.sel(**{Dimension.SAMPLE: valid})
        predictors = predictors.sel(**{Dimension.SAMPLE: valid})
        print('[INFO] Loading completed.')
        return predictors, targets


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
        ensemble = np.reshape(ensemble,(-1, shape[-1]))
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


def _load_forecasts_from_zarr(config: DataConfig, scalers, station_data):
    predictors, targets = ZarrLoader().load(config)
    predictors.load()
    targets.load()
    preprocess = Preprocess(config, scalers=scalers)
    data = preprocess.transform(predictors, targets, station_data, fit=False)
    return {'forecasts': data}, preprocess.scalers


def _load_data_from_zarr(config: DataConfig):
    print('[INFO] Loading reforcasts.')
    data, scalers, station_data = _load_reforecasts(config, return_station_data=True)
    print('[INFO] Loading forecasts.')
    forecasts, _ = _load_forecasts_from_zarr(config, scalers, station_data)
    data.update(forecasts)
    return data, scalers


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
    keys = ['train', 'val']
    if test:
        keys += ['test', 'forecasts']
    return tuple(data[key] for key in keys), conditions


def _test():
    config = DataConfig(6, 't2m', None, '12:4:4', 0.5, 'spherical')
    device = torch.device('cuda')
    data, conditions = build_datasets(config, cache_dir=TRAINING_CACHE_DIR, overwrite_cache=False, device=device)
    print('Done')
