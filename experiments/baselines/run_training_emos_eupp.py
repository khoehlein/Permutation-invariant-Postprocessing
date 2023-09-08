import argparse

import torch

import pandas as pd

from experiments.baselines.pp_emos import emos_pp

torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-train', type=str, required=True)
    parser.add_argument('--data-test', type=str, required=True)
    parser.add_argument('--output:path', type=str, default=None)
    parser.add_argument('--output:mode', type=str, default='exp',choices=['exp', 'run'])
    parser.add_argument('--training:ensemble-size', default=1, type=int)
    parser.add_argument('--training:num-epochs', default=150, type=int)
    parser.add_argument('--training:patience', default=10, type=int)
    parser.add_argument('--training:optimizer:lr', default=0.1, type=float)
    parser.add_argument('--model:local', action='store_true', dest='local_model')
    parser.add_argument('--model:init-sigma', type=float, default=0.)
    parser.add_argument('--model:posterior', type=str, default='logistic', choices=['logistic', 'normal'])
    parser.set_defaults(local_model=False)
    args = vars(parser.parse_args())

    df_train = pd.read_csv(args['data_train'], index_col=0)
    df_test = pd.read_csv(args['data_test'], index_col=0)

    pred_vars = ['ens_mean', 'ens_sd', 'month']


    pred = emos_pp(
        train=df_train,
        X=df_test,
        pred_vars=pred_vars,
        args={
            'n_sim': args['training:ensemble_size'],
            'lr_lbfgs': args['training:optimizer:lr'],
            'init_sigma': args['model:init_sigma'],
            'n_patience': args['training:patience'],
            'n_epochs': args['training:num_epochs'],
            'local': args['local_model'],
            'method': args['model:posterior']
        },
        output_path=args['output:path'],
        output_mode=args['output:mode'],
    )

    scores_pp = pred['scores_pp']
    print(scores_pp.describe())


if __name__ == '__main__':
    main()
