import argparse
import dataclasses
import datetime
import functools
import json
import os.path
import shutil
import uuid
from typing import Union, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
import xarray as xr
from torch import Tensor
from torch.utils.data import TensorDataset

from data.config.interface import read_path_configs, get_path
from data.euppbench.reforecasts import Preprocess as PreprocessBase


DEFAULT_SPLIT = '2015-01-01T00:2016-01-01T00:2017-01-01T00'


FORECAST_INIT_TIME = 0


DYNAMIC_PREDICTORS = [
    'VMAX_10M',  # 'VMAX_10M_LS', 'VMAX_10M_LS_S', 'VMAX_10M_MS', 'VMAX_10M_MS_S',
    'U_10M',  # 'U_10M_LS', 'U_10M_LS_S', 'U_10M_MS', 'U_10M_MS_S',
    'U500', 'U1000',  'U700', 'U850', 'U950',
    'V_10M',
    'V500', 'V1000', 'V700', 'V850', 'V950',
    'WIND_10M',
    'WIND500', 'WIND1000', 'WIND700', 'WIND850', 'WIND950',
    'OMEGA500', 'OMEGA1000',  'OMEGA700', 'OMEGA850', 'OMEGA950',
    'T_G',  # 'T_G_LS', 'T_G_LS_S', 'T_G_MS', 'T_G_MS_S',
    'T_2M',  # 'T_2M_LS', 'T_2M_LS_S', 'T_2M_MS', 'T_2M_MS_S',
    'T500', 'T1000','T700', 'T850', 'T950',
    'TD_2M', # 'TD_2M_LS', 'TD_2M_LS_S', 'TD_2M_MS', 'TD_2M_MS_S',
    'RELHUM500', 'RELHUM1000', 'RELHUM700', 'RELHUM850', 'RELHUM950',
    'TOT_PREC',
    'RAIN_GSP',
    'SNOW_GSP',
    'W_SNOW',
    'W_SO1', 'W_SO2', 'W_SO6', 'W_SO18', 'W_SO54',
    'CLCT',
    'CLCL',
    'CLCM',
    'CLCH',
    'HBAS_SC',
    'HTOP_SC',
    'ASOB_S',  # 'ASOB_S_LS', 'ASOB_S_LS_S', 'ASOB_S_MS', 'ASOB_S_MS_S',
    'ATHB_S',  # 'ATHB_S_LS', 'ATHB_S_LS_S', 'ATHB_S_MS', 'ATHB_S_MS_S',
    'ALB_RAD', # 'ALB_RAD_LS', 'ALB_RAD_LS_S', 'ALB_RAD_MS', 'ALB_RAD_MS_S',
    'PMSL', # 'PMSL_LS', 'PMSL_LS_S', 'PMSL_MS', 'PMSL_MS_S',
    'FI500', 'FI1000',  'FI700', 'FI850', 'FI950',
]

ENSEMBLE_SIZE = 20


@dataclasses.dataclass(unsafe_hash=True)
class DataConfig(object):
    flt: int
    target: str
    predictors: Union[None, str]
    splitting: str  # -> train, val, test
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
        group.add_argument('--data:target', type=str, help='names of target variable', default='wind_speed_of_gust')
        group.add_argument('--data:splitting', type=str, help='number of years for training, validation and test, separated by ":"', default=DEFAULT_SPLIT)
        group.add_argument('--data:location-embedding', type=str, help='transformfor lat-lon station coordinates', default='spherical', choices=['spherical', 'plain'])

    @classmethod
    def from_args(cls, args):
        config_dict = {
            'flt': args['data:flt'],
            'target': args['data:target'],
            'predictors': args['data:predictors'],
            'splitting': args['data:splitting'],
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


def _init_time_str(init_time:int):
    return '{:02d}'.format(init_time)


def get_raw_ensemble_base_path():
    return get_path('cosmo_de_ensemble')


def get_ensemble_base_path():
    return get_path('cosmo_de_reduced')


def get_observations_base_path():
    return get_path('cosmo_de_observations')


TRAINING_CACHE_DIR = get_path('cosmo_de_cache', assert_not_none=False)


@functools.lru_cache()
def load_station_data():
    loc_data_file = os.path.join(os.path.dirname(__file__), 'loc_data.csv')
    return pd.read_csv(loc_data_file, sep=',', index_col=0, encoding="ISO-8859-1")


def get_station_mapping():
    station_data = load_station_data()
    return {id: idx for idx, id in enumerate(station_data['station_id'])}


def add_winds(data):
    def add_wind_for(data, var_name):
        u = data[f'U{var_name}']
        v = data[f'V{var_name}']
        wind = np.sqrt(u ** 2 + v ** 2)
        data[f'WIND{var_name}'] = (list(wind.dims), wind.data)

    for var_name in ['500', '700', '850', '950', '1000', '_10M']:
        add_wind_for(data, var_name)


def _load_nc_file(path: str, station_data, config):
    base_path, file_name = os.path.split(path)
    time_stamp = np.datetime64(datetime.datetime.strptime(os.path.splitext(file_name)[0].split('_')[-1], '%Y%m%d%H'))
    data = xr.open_dataset(path, engine='netcdf4')
    station_mapping = {id: idx for idx, id in enumerate(data.data_vars['loc'].astype('S30').values)}
    station_idx = [station_mapping[id] for id in station_data['station_id'].values.astype('S30')]
    data = data.isel(station=station_idx)
    lead_time = data['step'].values.astype('timedelta64[h]')
    valid_time = time_stamp + lead_time
    data = data.drop_vars('step').rename_dims(fps='step', nensm='number').expand_dims(dim='time', axis=0)
    data = data.assign_coords({
        'lead_time': ('step', lead_time),
        'valid_time': ('step', valid_time),
        'init_time': ('time', [time_stamp])
    })
    data = data.set_index(step='lead_time')
    add_winds(data)
    data = data[config._predictor_names()]
    return data


def _load_raw_data(path: str, config: DataConfig):

    def is_valid_file(f: str):
        name, ext = os.path.splitext(f)
        return name.endswith(_init_time_str(0)) and ext == '.nc'

    station_data = load_station_data()
    files = [f for f in os.listdir(path) if is_valid_file(f)]
    data = []
    for file in files:
        data.append(_load_nc_file(os.path.join(path, file), station_data, config))
    return xr.concat(data, dim='time')


def reduce_raw_ensemble(config: DataConfig):
    raw_data_base_path = get_raw_ensemble_base_path()
    ensemble_base_path = get_ensemble_base_path()
    os.makedirs(ensemble_base_path, exist_ok=True)
    # station_data = load_station_data()
    month_dirs = [d for d in os.listdir(raw_data_base_path) if os.path.isdir(os.path.join(raw_data_base_path, d))]
    data = []
    with tqdm.tqdm(total=len(month_dirs)) as pbar:
        for month in sorted(month_dirs):
            data = _load_raw_data(os.path.join(raw_data_base_path, month), config)
            data.to_zarr(os.path.join(ensemble_base_path, month + '.zarr'))
            pbar.update(1)
    return data


class ZarrLoader(object):

    def _load_ensemble(self):
        path = get_ensemble_base_path()
        months = [f for f in sorted(os.listdir(path)) if f.endswith('.zarr')]
        data = [xr.open_zarr(os.path.join(path, f)) for f in months]
        data = xr.concat(data, dim='time')
        data = data.assign_coords(station_id=('station', load_station_data()['station_id'].values))
        return data

    def _load_observations(self, config: DataConfig):
        def is_valid_year(file_name):
            year = int(os.path.splitext(file_name)[0].split('-')[-1])
            return year >= 2010 and year < 2017
        path = get_observations_base_path()
        years = [f for f in sorted(os.listdir(path)) if f.endswith('.nc') and is_valid_year(f)]
        data = []
        station_data = load_station_data()
        for year_file in years:
            obs_data = xr.open_dataset(os.path.join(path, year_file))
            station_mapping = {id: idx for idx, id in enumerate(obs_data['station_id'].values.astype('S30'))}
            station_idx = [station_mapping[id] for id in station_data['station_id'].values.astype('S30') if id in station_mapping]
            obs_data = obs_data[config.target].isel(ncells=station_idx)
            valid = np.logical_and(
                obs_data.time >= np.datetime64('2010-12-08T00'),
                obs_data.time < np.datetime64('2017-01-01T00')
            )
            obs_data = obs_data.sel(time=valid)
            flt = np.timedelta64(config.flt, 'h')
            obs_data = obs_data.sel(time=(obs_data.time - obs_data.time.astype('datetime64[D]')) == flt)
            valid = [id in station_mapping for id in station_data['station_id'].values.astype('S30')]
            dummy = np.full((len(obs_data.time), len(station_data)), np.nan)
            dummy[:, valid] = obs_data.values
            obs_data = xr.DataArray(
                dummy,
                dims=['time', 'station'],
                coords={
                    # 'time': obs_data.time.values,
                    'valid_time': ('time', obs_data.time.values.astype('datetime64[h]')),
                    'station': station_data['station_id'].values,
                    'station_id': ('station', station_data['station_id'].values),
                },
                name=obs_data.name
            )
            data.append(obs_data)
        data = xr.concat(data, dim='time')
        return data

    @staticmethod
    def _find_valid_stations(data: xr.DataArray, completeness: float):
        availability = (1. - data.isnull().astype(np.float32)).mean('time')
        valid_stations = data['station_id'].sel(ncells=availability > completeness)
        return valid_stations

    @staticmethod
    def _get_yday(valid_time: xr.DataArray):
        year_lower = valid_time.values.astype('datetime64[h]').astype('datetime64[Y]')
        year_upper = (year_lower + np.timedelta64(1, 'Y')).astype('datetime64[h]')
        yday = (valid_time.astype('datetime64[h]') - year_lower) / (year_upper - year_lower.astype('datetime64[h]'))
        return yday

    @staticmethod
    def _extract_static_data(data: pd.DataFrame):
        lat = data['latitude'].values
        lon = data['longitude'].values
        alt = data['height'].values
        station_id = data['station_id'].values
        elv = alt - data['orog_DE'].values
        static_data = pd.DataFrame({
            'station_latitude': lat,
            'station_longitude': lon,
            'station_altitude': alt,
            'elevation_difference': elv,
            'station_id': station_id,
        })
        return static_data

    def load(self, config: DataConfig):
        flt = np.timedelta64(config.flt, 'h')
        print('[INFO] Loading data from disk.')
        targets = self._load_observations(config)
        predictors = self._load_ensemble().sel(step=flt)
        targets, predictors = self.merge_times(targets, predictors)
        predictors['yday'] = self._get_yday(targets['time'])
        station_data = self._extract_static_data(load_station_data())
        predictors = predictors.stack(sample=['time', 'station'])
        targets = targets.stack(sample=['time', 'station'])
        mask = np.full(predictors.dims['sample'], False)
        for key in predictors.data_vars:
            if 'number' in predictors[key].dims:
                mask = np.logical_or(mask, predictors[key].isnull().any(dim='number').values)
        valid = ~ np.logical_or(targets.isnull(), mask)
        targets = targets.sel(sample=valid)
        predictors = predictors.sel(sample=valid)
        print('[INFO] Loading completed.')
        return predictors, targets, station_data

    def merge_times(self, targets: xr.DataArray, predictors: xr.Dataset):
        target_times = targets['valid_time'].values
        predictor_times = predictors['valid_time'].values
        common_min = np.minimum(np.min(target_times), np.min(predictor_times))
        common_max = np.maximum(np.max(target_times), np.max(predictor_times))
        all_times = np.arange(common_min, common_max + np.timedelta64(24, 'h'), np.timedelta64(24, 'h')).astype('datetime64[h]')

        def get_mask(times):
            times = (times - common_min) / np.timedelta64(24, 'h')
            times = times.astype(int)
            mask = np.full_like(all_times, False)
            mask[times] = True
            return mask, times

        target_mask, target_idx = get_mask(target_times)
        predictor_mask, predictor_idx = get_mask(predictor_times)
        joint_mask = np.logical_and(target_mask, predictor_mask)
        joint_times = all_times[joint_mask]
        targets = targets.set_index(time='valid_time').sel(time=joint_times)
        predictors = predictors.set_index(time='valid_time').sel(time=joint_times)
        return targets, predictors


class Preprocess(PreprocessBase):

    PREDICTOR_TARGET_MAPPING = {'wind_speed_of_gust': 'VMAX_10M'}

    @staticmethod
    def _split_once(data: Union[xr.Dataset, xr.DataArray], split_idx: np.datetime64):
        lower = data['time'] < split_idx
        return data.sel(sample=lower), data.sel(sample=~lower)

    def split(self, dataset: xr.Dataset):
        max_train, max_val, max_test = [np.datetime64(s) for s in self.config.splitting.split(':')]
        dataset, _ = self._split_once(dataset, max_test)
        data_train, data_test = self._split_once(dataset, max_val)
        data_train, data_val = self._split_once(data_train, max_train)
        return {
            'train': data_train, 'val': data_val, 'test': data_test
        }


def _load_data_from_zarr(config: DataConfig):
    predictors, targets, station_data = ZarrLoader().load(config)
    predictors.load()
    targets.load()
    preprocess = Preprocess(config)
    data, targets = [preprocess.split(ds) for ds in [predictors, targets]]
    station_data = preprocess.compute_station_stats(data['train'], targets['train'], station_data)
    for key in ['train', 'val', 'test']:
        data[key] = preprocess.transform(data[key], targets[key], station_data, fit=(key == 'train'))
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


def build_training_dataset(args, device, test=False):
    config = DataConfig.from_args(args)
    data, conditions = build_datasets(config, cache_dir=TRAINING_CACHE_DIR, overwrite_cache=False, device=device)
    if test:
        return (data['train'], data['val'], data['test']), conditions
    return (data['train'], data['val']), conditions
