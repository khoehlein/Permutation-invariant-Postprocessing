import argparse
import os

from data.euppbench.forecasts import TRAINING_CACHE_DIR, ZarrLoader, Preprocess
from data.euppbench.reforecasts import (
    DataConfig,
    _load_data_from_zarr as _load_reforecasts
)


def export_csv():
    parser = argparse.ArgumentParser()
    DataConfig.init_parser(parser)
    parser.add_argument('--csv-directory', type=str, default=None)
    args = vars(parser.parse_args())
    config = DataConfig.from_args(args)

    csv_dir = args['csv_directory']
    if csv_dir is None:
        assert TRAINING_CACHE_DIR is not None
        csv_dir = os.path.abspath(os.path.join(TRAINING_CACHE_DIR, '../csv'))
    os.makedirs(csv_dir, exist_ok=True)

    print('[INFO] Loading reforecasts.')
    *_, station_data = _load_reforecasts(config, return_station_data=True)
    print('[INFO] Loading forecasts.')
    predictors, targets = ZarrLoader().load(config)
    print('[INFO] Creating test data frame.')
    preprocess = Preprocess(config)
    df = preprocess.to_data_frame(predictors, targets, station_data)
    print('[INFO] Writing tets data to CSV.')
    df.to_csv(os.path.join(csv_dir, f'df_test_51_flt-{config.flt}.csv'))
    print('[INFO] Done.')


if __name__ == '__main__':
    export_csv()
