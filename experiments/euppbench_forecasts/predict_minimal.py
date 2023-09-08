import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import tqdm

from data.euppbench.reforecasts import DYNAMIC_PREDICTORS as EUPP_PREDICTORS
from data.cosmo_de import DYNAMIC_PREDICTORS as GUSTS_PREDICTORS
from experiments.baselines.common import initiate, prepare_data, predict
from utils.automation.storage import MultiRunExperiment, PyTorchRun

N_ENS = 11

torch.set_num_threads(1)


def get_predictor_members(num_members, num_ref=11):
    p_ref = np.arange(1, num_ref + 1) / (num_ref + 1)
    members = np.round(p_ref * (num_members + 1))
    return [f'ens_{i}' for i in members.astype(int).tolist()]


def _get_channel_labels(ens_predictors, dynamic_predictors, target_predictor):
    return ens_predictors + \
    [p + '_mean' for p in dynamic_predictors if p != target_predictor] + \
    ['alt', 'orog', 'yday', 'loc_bias', 'loc_cover', 'lat', 'lon']


def get_channel_labels(dataset: str):
    predictors, target, num_ref = {
        'eupp': (EUPP_PREDICTORS, 't2m', 11),
        'gusts': (GUSTS_PREDICTORS, 'VMAX_10M', 20),
    }[dataset]
    return _get_channel_labels(['ens_mean', 'ens_sd'], predictors, target)


class MinimalPredict(object):

    def __init__(
            self,
            data_train: str,
            data_test: str,
            dynamic_predictors: List[str],
            target_predictor: str,
            reference_ensemble_size: int,
            ensemble_mode=None,
    ):
        self.df_train, ens_predictors = self.read_csv(data_train, ensemble_mode)
        self.df_test, _ = self.read_csv(data_test, ensemble_mode)
        self.reference_ensemble_size = reference_ensemble_size
        self.pred_vars = _get_channel_labels(ens_predictors, dynamic_predictors, target_predictor)
        # self.pred_vars = [p + '_mean' for p in DYNAMIC_PREDICTORS] + ['alt', 'orog', 'yday', 'loc_bias', 'loc_cover', 'lat', 'lon']
        self.loc_id_vec = np.sort(np.unique(self.df_train['location'].values))

    def read_csv(self, file_path: str, ensemble_mode):
        data = pd.read_csv(file_path, index_col=0)

        def is_ens_predictor(s: str):
            s = str(s)
            return s.startswith('ens') and s[-1].isnumeric()

        if ensemble_mode is not None:
            ensemble_predictors = [c for c in data.columns if is_ens_predictor(c)]
            if ensemble_mode == 'random':
                ensemble_values = data.loc[:, ensemble_predictors].values
                gen = np.random.Generator(np.random.PCG64(42))
                samples = gen.random(size=ensemble_values.shape)
                samples = np.argsort(samples, axis=-1)[:, :11]
                samples = np.sort(ensemble_values[np.arange(len(data))[:, None], samples], axis=-1)
                # predictor_names = [f'ens_pred_{i + 1}' for i in range(11)]
            elif ensemble_mode == 'quantile':
                ensemble_size = len(ensemble_predictors)
                predictor_names = get_predictor_members(ensemble_size, num_ref=self.reference_ensemble_size)
                samples = np.sort(data.loc[:, predictor_names].values, axis=-1)
                # predictors = data.loc[:, predictor_names].rename(columns={p: f'ens_pred_{i + 1}' for i, p in enumerate(predictor_names)})
            else:
                raise ValueError(f'[ERROR] Unknown ensemble mode: {ensemble_mode}')
            predictor_names = [f'ens_pred_{i + 1}' for i in range(11)]
            predictors = pd.DataFrame(data=samples, index=data.index, columns=predictor_names)
            data = pd.concat([data, predictors], axis=1)
        else:
            predictor_names = ['ens_mean', 'ens_sd']
        return data, predictor_names


    def load_data(self, run):
        X, nn_ls, pred_vars, train = initiate(
            self.df_test, None, N_ENS, None, self.pred_vars, False, True, self.df_train
        )
        (
            X_pred,
            X_train, y_train,
            X_valid, y_valid,
            n_dir_preds, n_test, n_train, n_valid
        ) = prepare_data(
            X, None, self.loc_id_vec, self.pred_vars, train
        )
        checkpoints = run.list_checkpoints(sort_output=True)
        return X_pred, checkpoints

    def from_run_path(self, run_path: str):
        # access storage
        experiment, run = MultiRunExperiment.from_run_path(run_path, return_run=True, _except_on_not_existing=True)
        device = torch.device('cpu')

        # load data
        X_pred, checkpoints, obs = self.load_data(run)

        # prepare output directory
        output_path = os.path.abspath(os.path.join(run.get_evaluation_path(), 'predictions', 'forecasts'))
        os.makedirs(output_path, exist_ok=True)

        print(f'[INFO] Predicting for run {run.path}.')
        with torch.no_grad():
            with tqdm.tqdm(total=len(checkpoints)) as pbar:
                for i, checkpoint in enumerate(checkpoints):
                    model = run.load_checkpoint(checkpoint, map_location=device)
                    predictions = predict(model, X_pred, batch_size=256, verbose=False).data.cpu().numpy()
                    if predictions.shape[-1] > 2:
                        # bqn case
                        predictions = np.cumsum(predictions, axis=-1)
                    np.savez(
                        os.path.join(
                            output_path,
                            os.path.splitext(checkpoint)[0] + '.npz'
                        ),
                        predictions=predictions,
                        obs=self.df_test['obs'].values,
                    )
                    pbar.update(1)

        return None

    def from_experiment_path(self, path: str):
        experiment = MultiRunExperiment(path)

        print(f'[INFO] Predicting for experiment directory at {experiment.path}')

        def predict_for_run(run: PyTorchRun):
            return self.from_run_path(run.path)

        experiment.map_across_runs(predict_for_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--data-train', type=str, required=True)
    parser.add_argument('--data-test', type=str, required=True)
    parser.add_argument('--ensemble', type=str, default=None)
    parser.add_argument('--dataset',type=str, default='eupp', choices=['eupp', 'gusts'])
    parser.set_defaults(recurse=False)
    args = vars(parser.parse_args())

    print(f'[INFO] Predicting for dataset "forecasts"')
    predictors, target, num_ref = {
        'eupp': (EUPP_PREDICTORS, 't2m', 11),
        'gusts': (GUSTS_PREDICTORS, 'VMAX_10M', 20),
    }[args['dataset']]
    predict = MinimalPredict(
        args['data_train'],
        args['data_test'],
        predictors, target, num_ref,
        ensemble_mode=args['ensemble']
    )
    predict.from_experiment_path(args['path'])


if __name__ == '__main__':
    main()
