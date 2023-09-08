import argparse
import os.path

import numpy as np
import torch
import tqdm

from data.utils import BatchLoader
from data.euppbench.reforecasts import build_training_dataset as build_eupp_data
from experiments.common_utils import prepare_data
from model.loss import factory
from utils.automation.storage import MultiRunExperiment, PyTorchRun


torch.set_num_threads(4)

class Predict(object):

    def __init__(self, dataset: str):
        self.dataset = dataset

    def from_run_path(self, run_path: str, device=None):
        if device is None:
            device = torch.device('cpu')
        experiment, run = MultiRunExperiment.from_run_path(run_path, return_run=True)
        args = run.parameters()
        checkpoints = run.list_checkpoints(sort_output=True)
        loss = factory.build_loss(args)
        loader, conditions = self._prepare_data_loader(args, device)
        output_path = self._prepare_output_path(run)
        print('[INFO] Predicting.')
        with tqdm.tqdm(total=len(checkpoints)) as pbar:
            for i, checkpoint in enumerate(checkpoints):
                model = run.load_checkpoint(checkpoint, map_location=device)
                all_obs, all_preds = self._compute_predictions(model, loader, conditions, loss, device)
                checkpoint_output_file = os.path.join(output_path, os.path.splitext(checkpoint)[0] + '.npz')
                self._export_predictions(checkpoint_output_file, all_preds, all_obs)
                pbar.update(1)

    def _prepare_output_path(self, run: PyTorchRun, *args, **kwargs):
        output_path = os.path.abspath(os.path.join(run.get_evaluation_path(), 'predictions', self.dataset))
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def _compute_predictions(self, model, loader, conditions, loss, device):
        all_preds = None
        all_obs = np.zeros((len(loader.dataset),))
        i = 0
        for batch in iter(loader):
            predictors, observations = prepare_data(batch, conditions, model, device)
            num_samples = len(observations)
            predictions = model['model'](*predictors)
            params = loss.compute_parameters(predictions, merge=True)
            if all_preds is None:
                all_preds = np.zeros((len(loader.dataset), params.shape[-1]))
            all_preds[i:(i + num_samples)] = params.data.cpu().numpy()
            all_obs[i:(i + num_samples)] = observations.data.cpu().numpy().ravel()
            i = i + num_samples
        return all_obs, all_preds

    def _export_predictions(self, output_path, all_preds, all_obs):
        np.savez(output_path, predictions=all_preds, obs=all_obs)

    def _prepare_data_loader(self, args, device):
        data, conditions = self._load_data(args, device)
        loader = BatchLoader(data, batch_size=128*args['training:batch_size'], shuffle=False, drop_last=False)
        return loader, conditions

    def _load_data(self, args, device):
        data, conditions = build_eupp_data(args, test=True, device=device)
        data = {key: val for key, val in zip(['train', 'valid', 'test'], data)}[self.dataset]
        return data, conditions

    def from_experiment_path(self, path: str, device=None):
        experiment = MultiRunExperiment(path)

        print(f'[INFO] Predicting for experiment directory at {experiment.path}')

        def predict_for_run(run: PyTorchRun):
            return self.from_run_path(run.path, device=device)

        experiment.map_across_runs(predict_for_run)


    def predict_or_recurse(self, path, max_depth=None, device=None):
        try:
            self.from_experiment_path(path, device=device)
            success = True
        except AssertionError:
            success = False
        if not success:
            print('[INFO] Recursing...')
            if max_depth is None or max_depth > 0:
                contents = [c for c in os.listdir(path) if os.path.isdir(os.path.join(path, c))]
                new_max_depth = max_depth - 1 if max_depth is not None else None
                for c in sorted(contents):
                    self.predict_or_recurse(os.path.join(path, c), max_depth=new_max_depth, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('-r', action='store_true', dest='recurse')
    parser.set_defaults(recurse=False)
    args = vars(parser.parse_args())

    with torch.no_grad():
        for dataset in ['test', 'valid']:
            print(f'[INFO] Predicting for dataset "{dataset}"')
            predict = Predict(dataset)
            if args['recurse']:
                predict.predict_or_recurse(args['path'])
            else:
                predict.from_run_path(args['path'])


if __name__ == '__main__':
    main()
