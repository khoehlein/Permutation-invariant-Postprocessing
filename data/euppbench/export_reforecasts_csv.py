import argparse
import os

import numpy as np

from data.euppbench.reforecasts import DataConfig, TRAINING_CACHE_DIR, ZarrLoader, Preprocess


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

    predictors, targets, station_data = ZarrLoader().load(config)
    preprocess = Preprocess(config)
    max_train, max_val, max_test = np.cumsum([int(s) for s in config.splitting.split(':')])
    predictors, _ = preprocess._split_once(predictors, max_test)
    targets, _ = preprocess._split_once(targets, max_test)
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
