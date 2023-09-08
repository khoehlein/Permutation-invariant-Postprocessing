import argparse
import os

import xarray as xr
import climetlab as cml


COUNTRIES = [
    'austria',
    'germany',
    'netherlands',
    'belgium',
    'france',
]

MODE = 'forecasts'

DATASET_KEY = f'EUPPBench-training-data-stations-{MODE}'

def load_surface_dataset(key, country):
    if MODE == 'reforecasts':
        return cml.load_dataset(key, country)
    elif MODE == 'forecasts':
        return cml.load_dataset(key, 'ensemble', country)
    else:
        raise NotImplementedError()


def load_level_dataset(key, level, country):
    if MODE == 'reforecasts':
        return cml.load_dataset(key, level, country)
    elif MODE == 'forecasts':
        return cml.load_dataset(key, level, 'ensemble', country)
    else:
        raise NotImplementedError()

VARIABLE_GROUPS = [
    'surface',
    'surface-processed',
    'pressure',
]

PRESSURE_LEVELS = [500, 700, 850]

STATIC_FIELDS = ['landu', 'mterh', 'z']


def export_data(data: xr.Dataset, file_name: str):
    print('[INFO] Exporting.')
    encoding = None
    data.to_zarr(file_name, encoding=encoding)
    print('[INFO] Done.')


def load_static_variables(output_path):
    print('[INFO] Loading static data')
    data = xr.Dataset(
        data_vars={
            field: cml.load_dataset('EUPPBench-training-data-gridded-static-fields', field).to_xarray()[field]
            for field in STATIC_FIELDS
        },
    )
    export_data(
        data,
        os.path.join(output_path, 'static_fields.zarr')
    )


def load_station_data(output_base_path: str):
    for country in COUNTRIES:
        print(f'[INFO] Processing country {country}.')
        for variable_group in VARIABLE_GROUPS:
            group_key = '-'.join([DATASET_KEY, variable_group])
            group_output_path = os.path.join(output_base_path, variable_group)
            print(f'[INFO] Loading {variable_group} data.')
            if variable_group == 'pressure':
                for level in PRESSURE_LEVELS:
                    level_output_path = os.path.join(group_output_path, str(level))
                    os.makedirs(level_output_path, exist_ok=True)
                    dataset = load_level_dataset(group_key, level, country)
                    export_data(
                        dataset.to_xarray(),
                        os.path.join(level_output_path, f'{country}.zarr')
                    )
            elif variable_group.startswith('surface'):
                dataset = load_surface_dataset(group_key, country)
                export_data(
                    dataset.to_xarray(),
                    os.path.join(group_output_path, f'{country}.zarr')
                )
                group_output_path = os.path.join(output_base_path, variable_group.replace('surface', 'observations'))
                os.makedirs(group_output_path, exist_ok=True)
                export_data(
                    dataset.get_observations_as_xarray(),
                    os.path.join(group_output_path, f'{country}.zarr')
                )
            else:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = vars(parser.parse_args())

    output_base_path = os.path.abspath(args['path'])
    os.makedirs(output_base_path, exist_ok=True)

    load_station_data(output_base_path)
    load_static_variables(output_base_path)


if __name__ == '__main__':
    main()
