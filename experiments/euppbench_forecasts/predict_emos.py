import argparse
import os

import numpy as np
import pandas as pd
import torch
import tqdm

from experiments.baselines.common import initiate, prepare_data
from experiments.baselines.pp_emos import predict
from utils.automation.storage import MultiRunExperiment, PyTorchRun

N_ENS = 11

torch.set_num_threads(1)

class EMOSPredict(object):

    def __init__(
            self,
            data_train: str,
            data_test: str,
    ):
        self.df_train = pd.read_csv(data_train, index_col=0)
        self.df_test = pd.read_csv(data_test, index_col=0)
        self.pred_vars = ['ens_mean', 'ens_sd', 'month']
        self.loc_id_vec = np.sort(np.unique(self.df_train['location'].values))

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
            X, None, self.loc_id_vec, self.pred_vars, train, rescale=False
        )
        checkpoints = run.list_checkpoints(sort_output=True)
        return X_pred, checkpoints

    def from_run_path(self, run_path: str):
        # access storage
        experiment, run = MultiRunExperiment.from_run_path(run_path, return_run=True, _except_on_not_existing=True)
        device = torch.device('cpu')

        # load data
        X_pred, checkpoints = self.load_data(run)

        # prepare output directory
        output_path = os.path.abspath(os.path.join(run.get_evaluation_path(), 'predictions', 'forecasts'))
        os.makedirs(output_path, exist_ok=True)

        print(f'[INFO] Predicting for run {run.path}.')
        with torch.no_grad():
            with tqdm.tqdm(total=len(checkpoints)) as pbar:
                for i, checkpoint in enumerate(checkpoints):
                    model = run.load_checkpoint(checkpoint, map_location=device)
                    predictions = predict(model, X_pred)
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
    parser.set_defaults(recurse=False)
    args = vars(parser.parse_args())

    print(f'[INFO] Predicting for dataset "forecasts"')
    predict = EMOSPredict(
        args['data_train'],
        args['data_test'],
    )
    predict.from_experiment_path(args['path'])


if __name__ == '__main__':
    main()
