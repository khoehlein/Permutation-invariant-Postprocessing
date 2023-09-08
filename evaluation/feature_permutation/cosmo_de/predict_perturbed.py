import os

import torch
import tqdm

from evaluation.feature_permutation.perturbations import (
    PerturbationFactory, EnsembleRankShuffle, MetricPreservationShuffle
)
from evaluation.feature_permutation.euppbench.predict_perturbed import PredictPerturbed as BasePredictPerturbed
from data.cosmo_de import DataConfig
from utils.automation.storage import MultiRunExperiment
import argparse
from model.loss import factory
from data.cosmo_de import build_training_dataset as build_gusts_data

torch.set_num_threads(6)


def get_channel_labels(args):
    config = DataConfig.from_args(args)
    return config._predictor_names()


class PredictPerturbed(BasePredictPerturbed):

    def from_run_path(self, run_path: str, device=None):
        if device is None:
            device = torch.device('cpu')
        experiment, run = MultiRunExperiment.from_run_path(run_path, return_run=True, _except_on_not_existing=True)
        args = run.parameters()
        checkpoints = run.list_checkpoints(sort_output=True)
        loss = factory.build_loss(args)
        num_channels = len(get_channel_labels(args))
        for i in range(num_channels):
            perturbation = self.perturbation_factory.generate(num_channels, [i])
            loader, conditions = self._prepare_data_loader(args, device)
            perturbation.perturb_dataset(loader.dataset)
            output_path = self._prepare_output_path(run, perturbation)
            print(f'[INFO] Predicting for channel {i + 1} of {num_channels}.')

            with tqdm.tqdm(total=len(checkpoints)) as pbar:
                for i, checkpoint in enumerate(checkpoints):
                    model = run.load_checkpoint(checkpoint, map_location=device)
                    all_obs, all_preds = self._compute_predictions(model, loader, conditions, loss, device)
                    checkpoint_output_file = os.path.join(output_path, os.path.splitext(checkpoint)[0] + '.npz')
                    self._export_predictions(checkpoint_output_file, all_preds, all_obs)
                    pbar.update(1)

    def _load_data(self, args, device):
        data, conditions = build_gusts_data(args, test=True, device=device)
        data = {key: val for key, val in zip(['train', 'valid', 'test'], data)}[self.dataset]
        return data, conditions


def main_v2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--num-bins', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-r', action='store_true', dest='recurse')
    parser.add_argument('--gpu', action='store_true', dest='use_gpu')
    parser.set_defaults(recurse=False, use_gpu=False)
    args = vars(parser.parse_args())

    device = torch.device('cuda:0' if args['use_gpu'] else 'cpu')

    with torch.no_grad():
        for dataset in ['test']:
            perturbation_factory = PerturbationFactory(
                EnsembleRankShuffle,
                seed=args['seed']
            )
            print(f'[INFO] Predicting for dataset "{dataset}"')
            print(f'[INFO] Rank perturbation')
            predict = PredictPerturbed(dataset, perturbation_factory)
            if args['recurse']:
                predict.predict_or_recurse(args['path'], device=device)
            else:
                predict.from_run_path(args['path'], device=device)
            for metric_name in MetricPreservationShuffle.available_metrics():
                print(f'[INFO] Metric perturbation: {metric_name}')
                for rerank in [True]:
                    perturbation_factory = PerturbationFactory(
                        MetricPreservationShuffle,
                        num_bins=args['num_bins'], metric=metric_name,
                        preserve_ranking=rerank, seed=args['seed']
                    )
                    predict = PredictPerturbed(dataset, perturbation_factory)
                    if args['recurse']:
                        predict.predict_or_recurse(args['path'], device=device)
                    else:
                        predict.from_run_path(args['path'], device=device)
            for rerank in [True, False]:
                perturbation_factory = PerturbationFactory(
                    MetricPreservationShuffle,
                    num_bins=1, metric='mean',
                    preserve_ranking=rerank, seed=args['seed']
                )
                predict = PredictPerturbed(dataset, perturbation_factory)
                if args['recurse']:
                    predict.predict_or_recurse(args['path'], device=device)
                else:
                    predict.from_run_path(args['path'], device=device)

if __name__ == '__main__':
    main_v2()
