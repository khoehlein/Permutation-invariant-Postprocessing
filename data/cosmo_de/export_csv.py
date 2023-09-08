import argparse
import os

import numpy as np

from data.cosmo_de import DataConfig, get_ensemble_base_path, ZarrLoader, Preprocess


def export_csv():
    parser = argparse.ArgumentParser()
    DataConfig.init_parser(parser)
    parser.add_argument('--csv-path', type=str, default=None)
    args = vars(parser.parse_args())
    config = DataConfig.from_args(args)

    csv_dir = args['csv_path']
    if csv_dir is None:
        csv_dir = os.path.abspath(os.path.join(get_ensemble_base_path(), '../csv'))
    os.makedirs(csv_dir, exist_ok=True)

    predictors, targets, station_data = ZarrLoader().load(config)
    preprocess = Preprocess(config)
    max_train, max_val, max_test = [np.datetime64(s) for s in config.splitting.split(':')]
    X_train, X_test = preprocess._split_once(predictors, max_val)
    Y_train, Y_test = preprocess._split_once(targets, max_val)
    station_data = preprocess.compute_station_stats(X_train, Y_train, station_data)
    print('[INFO] Creating training data frame.')
    df = preprocess.to_data_frame(X_train, Y_train, station_data)
    print('[INFO] Writing training data to CSV.')
    df.to_csv(os.path.join(csv_dir, f'df_train_flt-{config.flt}.csv'))
    print('[INFO] Creating test data frame.')
    df = preprocess.to_data_frame(X_test, Y_test, station_data)
    print('[INFO] Writing test data to CSV.')
    df.to_csv(os.path.join(csv_dir, f'df_test_flt-{config.flt}.csv'))
    print('[INFO] Done.')


if __name__ == '__main__':
    export_csv()
