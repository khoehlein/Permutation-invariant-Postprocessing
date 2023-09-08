import argparse
import os

import numpy as np

os.environ['R_HOME'] = os.path.expanduser('~/anaconda3/envs/rpy2/lib/R')

import pandas as pd
# import pyreadr

import rpy2
from rpy2 import robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()


def rpy2_read_r(r_path, df_name):
    out = robjects.r['get'](robjects.r['load'](r_path))
    print('Loading complete')
    df = pandas2ri.rpy2py(out)
    return df


def rdata_to_csv(df_name, r_path, csv_path):
    print('[INFO] Reading RData.')
    # df = pyreadr.read_r(r_path)[df_name]
    df = rpy2_read_r(r_path, df_name)
    print(df.head())
    print('[INFO] Writing CSV.')
    df.to_csv(csv_path)


def rpy2_write_r(r_path, df, df_name=None):
    r_data = pandas2ri.py2rpy(df)
    robjects.r.assign(df_name, r_data)
    robjects.r(f"save({df_name}, file='{r_path}')")


def csv_to_rdata(df_name, csv_path, r_path):
    print('[INFO] Reading CSV.')
    df = pd.read_csv(csv_path, index_col=0)
    print(df.head())
    print('[INFO] Writing RData.')
    rpy2_write_r(r_path, df, df_name=df_name)
    # pyreadr.write_rdata(r_path, df, df_name=df_name)


def _get_output_file_name(out_file: str, in_file: str):
    assert in_file is not None, '[ERROR] Input file must not be None!'
    if out_file is not None:
        return out_file
    in_file_name, in_file_ext = os.path.splitext(os.path.split(in_file)[-1])
    out_file_ext = {'.csv': '.Rdata', '.RData': '.csv'}[in_file_ext]
    out_file = in_file_name + out_file_ext
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, help='command what to do', choices=['r2csv', 'csv2r', 'rwr'])
    parser.add_argument('--csv', type=str, help='path to csv file', default=None)
    parser.add_argument('--rdata',type=str, help='path to rdata file', default=None)
    parser.add_argument('--df-name',type=str, help='nameof df to read from or write to rdata file')
    args = vars(parser.parse_args())

    csv_path = args['csv']
    rdata_path = args['rdata']
    df_name = args['df_name']
    action = args['action']

    if action == 'csv2r':
        rdata_path = _get_output_file_name(rdata_path, csv_path)
        csv_to_rdata(df_name, csv_path, rdata_path)
    elif action == 'r2csv':
        csv_path = _get_output_file_name(csv_path, rdata_path)
        rdata_to_csv(df_name, rdata_path, csv_path)
    elif action == 'rwr':
        csv_path = _get_output_file_name(csv_path, rdata_path)
        rdata_to_csv(df_name, rdata_path, csv_path)
        rdata_path = os.path.splitext(rdata_path)[0] + '_new' + '.RData'
        csv_to_rdata(df_name, csv_path, rdata_path)
        print('[INFO] Reading RData')
        df = rpy2_read_r(rdata_path, df_name)
        print(df.head())
    else:
        raise Exception(f'[ERROR] Unknown action: {action}')


if __name__ == '__main__':
    main()
