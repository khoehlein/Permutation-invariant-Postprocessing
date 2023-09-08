import argparse

import numpy as np
import torch

from data.euppbench.reforecasts import DYNAMIC_PREDICTORS, ENSEMBLE_SIZE
from experiments.baselines.pp_drn import drn_pp
import pandas as pd


torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-train', type=str, required=True)
    parser.add_argument('--data-test', type=str, required=True)
    parser.add_argument('--output:path', type=str, default=None)
    parser.add_argument('--output:mode', type=str, default='exp',choices=['exp', 'run'])
    parser.add_argument('--training:ensemble-size', default=20, type=int)
    parser.add_argument('--training:num-epochs', default=150, type=int)
    parser.add_argument('--training:patience', default=10, type=int)
    parser.add_argument('--training:optimizer:lr', default=5.e-4, type=float)
    parser.add_argument('--model:station-embedding', default=10, type=int)
    parser.add_argument('--model:num-channels', default=64, type=int)
    parser.add_argument('--model:posterior', type=str, default='logistic', choices=['logistic', 'normal'])
    args = vars(parser.parse_args())

    df_train = pd.read_csv(args['data_train'], index_col=0)
    df_test = pd.read_csv(args['data_test'], index_col=0)

    pred_vars = ['ens_mean', 'ens_sd'] + \
                [p + '_mean' for p in DYNAMIC_PREDICTORS if p != 't2m'] + \
                ['alt', 'orog', 'yday', 'loc_bias', 'loc_cover', 'lat', 'lon']

    loc_id_vec = np.sort(np.unique(df_train['location'].values))

    pred = drn_pp(
        train=df_train,
        X=df_test,
        i_valid=None,
        loc_id_vec=loc_id_vec,
        pred_vars=pred_vars,
        nn_ls={
            'n_sim': args['training:ensemble_size'],
            'lr_adam': args['training:optimizer:lr'],
            'n_patience': args['training:patience'],
            'emb_dim': args['model:station_embedding'],
            'lay1': args['model:num_channels'],
            'nn_verbose': True,
            'n_epochs': args['training:num_epochs'],
            'method': args['model:posterior'],
        },
        n_ens=ENSEMBLE_SIZE,
        output_path=args['output:path'],
        output_mode=args['output:mode'],
    )

    scores_pp: pd.Series = pred['scores_pp']
    print(scores_pp.describe())


if __name__ == '__main__':
    main()
