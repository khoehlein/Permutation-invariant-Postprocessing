import os

import torch
import tqdm

from evaluation.feature_permutation.perturbations import PerturbationFactory, LocationScaleShuffle, \
    EnsembleRankShuffle, MetricPreservationShuffle
from experiments.euppbench_reforecasts.predict import Predict as BasePredict
from data.euppbench.reforecasts import DataConfig
from utils.automation.storage import PyTorchRun, MultiRunExperiment
import argparse
from model.loss import factory

torch.set_num_threads(6)


def get_channel_labels(args):
    config = DataConfig.from_args(args)
    return config._predictor_names()


class PredictPerturbed(BasePredict):

    def __init__(self, dataset: str, perturbation_factory: PerturbationFactory):
        super().__init__(dataset)
        self.perturbation_factory = perturbation_factory

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

def build_perturbation_factory(args):
    if args['loc_bins'] is not None or args['scale_bins'] is not None:
        f = PerturbationFactory(
            LocationScaleShuffle,
            num_loc_bins=args['loc_bins'], num_scale_bins=args['scale_bins'],
            loc_metric=args['loc_metric'], scale_metric=args['scale_metric'],
            preserve_ranking=args['preserve_ranking'], seed=args['seed']
        )
    else:
        assert not args['preserve_ranking'], '[ERROR] Current settings do not admit shuffling.'
        f = PerturbationFactory(
            EnsembleRankShuffle,
            seed=args['seed']
        )
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--loc-bins', type=int, default=None)
    parser.add_argument('--scale-bins', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--loc-metric', type=str, default='mean')
    parser.add_argument('--scale-metric', type=str, default='std')
    parser.add_argument('--rerank', action='store_true', dest='preserve_ranking')
    parser.add_argument('-r', action='store_true', dest='recurse')
    parser.add_argument('--gpu', action='store_true', dest='use_gpu')
    parser.set_defaults(recurse=False, preserve_ranking=False, use_gpu=False)
    args = vars(parser.parse_args())

    perturbation_factory = build_perturbation_factory(args)
    device = torch.device('cuda:0' if args['use_gpu'] else 'cpu')

    with torch.no_grad():
        for dataset in ['test', 'valid']:
            print(f'[INFO] Predicting for dataset "{dataset}"')
            predict = PredictPerturbed(dataset, perturbation_factory)
            if args['recurse']:
                predict.predict_or_recurse(args['path'], device=device)
            else:
                predict.from_run_path(args['path'], device=device)


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
