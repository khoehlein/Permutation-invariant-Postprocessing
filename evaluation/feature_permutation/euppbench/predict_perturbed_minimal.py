import argparse
import os
from typing import List

import numpy as np
import torch
import tqdm

from evaluation.feature_permutation.perturbations import PerturbationFactory, ScalarPredictorPermutation
from experiments.euppbench_forecasts.predict_minimal import MinimalPredict
from experiments.baselines.common import predict
from utils.automation.storage import MultiRunExperiment, PyTorchRun

from data.euppbench.reforecasts import DYNAMIC_PREDICTORS as EUPP_PREDICTORS
from data.cosmo_de import DYNAMIC_PREDICTORS as GUSTS_PREDICTORS


class MinimalPredictPerturbed(MinimalPredict):

    def __init__(
            self,
            data_train: str,
            data_test: str,
            dynamic_predictors: List[str],
            target_predictor: str,
            reference_ensemble_size: int,
            dataset: str,
            perturbation_factory:PerturbationFactory,
            ensemble_mode=None,
    ):
        super().__init__(data_train, data_test, dynamic_predictors, target_predictor, reference_ensemble_size, ensemble_mode)
        self.dataset = dataset
        self.perturbation_factory = perturbation_factory

    def from_run_path(self, run_path: str):
        device = torch.device('cpu')
        experiment, run = MultiRunExperiment.from_run_path(run_path, return_run=True, _except_on_not_existing=True)
        num_channels = len(self.pred_vars)

        for i in range(num_channels):
            X_pred, checkpoints = self.load_data(run)
            perturbation = self.perturbation_factory.generate(num_channels, [i])
            perturbation.perturb_dataset(X_pred)
            output_path = self._prepare_output_path(run, perturbation)
            print(f'[INFO] Predicting for channel {i + 1} of {num_channels}.')

            with tqdm.tqdm(total=len(checkpoints)) as pbar:
                for i, checkpoint in enumerate(checkpoints):
                    model = run.load_checkpoint(checkpoint, map_location=device)
                    predictions = predict(model, X_pred, batch_size=256, verbose=False).data.cpu().numpy()
                    if predictions.shape[-1] > 2:
                        # bqn case
                        predictions = np.cumsum(predictions, axis=-1)
                    checkpoint_output_file = os.path.join(output_path, os.path.splitext(checkpoint)[0] + '.npz')
                    np.savez(checkpoint_output_file, predictions=predictions, obs=self.df_test['obs'].values)
                    pbar.update(1)

    def _prepare_output_path(self, run: PyTorchRun, *args, **kwargs):
        perturbation = args[0]
        output_path = os.path.abspath(
            os.path.join(
                run.get_evaluation_path(),
                'predictions',
                'perturbed',
                perturbation.name,
                self.dataset
            )
        )
        os.makedirs(output_path, exist_ok=True)
        return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--data-train', type=str, required=True)
    parser.add_argument('--data-test', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ensemble', type=str, default=None)
    parser.add_argument('--dataset', type=str, required=True, choices=['gusts', 'eupp-test', 'eupp-forecasts'])
    args = vars(parser.parse_args())

    perturbation_factory = PerturbationFactory(ScalarPredictorPermutation, seed=args['seed'])

    dataset = args['dataset'].split('-')
    if len(dataset) == 1:
        dataset = dataset[0]
        datafold = 'test'
    else:
        dataset, datafold = dataset

    print(f'[INFO] Predicting for dataset "{datafold}"')

    predictors, target, num_ref = {
        'eupp': (EUPP_PREDICTORS, 't2m', 11),
        'gusts': (GUSTS_PREDICTORS, 'VMAX_10M', 20),
    }[dataset]
    predict = MinimalPredictPerturbed(
        args['data_train'], args['data_test'],
        predictors, target, num_ref,
        datafold, perturbation_factory,
        ensemble_mode=args['ensemble']
    )
    predict.from_experiment_path(args['path'])


if __name__ == '__main__':
    main()
