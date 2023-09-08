import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation.metrics import compute_metrics_logistic, compute_metrics_bqn, compute_metrics_normal
from utils.automation.storage import MultiRunExperiment, PyTorchRun

SEED = 42
ENSEMBLE_SIZE = 10 # number of models in ensemble
gen = np.random.Generator(np.random.PCG64(SEED))


class Eval(object):

    def __init__(self, mode: str, model: str, num_members: int,  num_processes=8):
        self.mode = mode
        self.model = model
        self.num_members = num_members
        self.num_processes = num_processes

    def eval_predictions(self, y_pred: np.ndarray, y_true: np.ndarray):
        if self.model == 'bqn':
            return compute_metrics_bqn(y_pred, y_true, n_ens=self.num_members)
        elif self.model == 'logistic':
            return compute_metrics_logistic(y_pred, y_true, n_ens=self.num_members)
        elif self.model == 'normal':
            return compute_metrics_normal(y_pred, y_true, n_ens=self.num_members)
        else:
            raise Exception()


    def _eval_parallel(self, x):
        return self.eval_predictions(*x)


    def sample_ensemble_prediction(self, predictions):
        num_members = len(predictions)
        sample = gen.choice(np.arange(num_members), size=(ENSEMBLE_SIZE,), replace=False)
        return np.mean(predictions[sample], axis=0)


    def eval_retrain(self, base_dir: str):
        contents = [c for c in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, c))]
        try:
            params = pd.DataFrame([
                {key: int(val) for key, val in [f.split('-') for f in d.split('_')]}
                for d in contents
            ])
        except ValueError:
            params = None
        else:
            params['path'] = [base_dir]* len(params)
            params.to_csv(os.path.join(base_dir, 'params.csv'))

        def get_npz_path(d):
            experiment = MultiRunExperiment(os.path.join(base_dir, d))
            run_names = experiment.list_runs()
            assert len(run_names) == 1
            run = experiment.load_run(run_names[0])
            return os.path.join(run.get_evaluation_path(), 'predictions', self.mode)

        results = [
            self.eval_npz(get_npz_path(d))
            for d in contents
        ]
        results = pd.concat(results, axis=0)
        results.to_csv(os.path.join(base_dir, 'results.csv'))
        return params, results


    def eval_npz(self, base_dir: str):
        pd.options.display.max_columns = 10
        pd.options.display.width = 1024
        files = sorted([f for f in os.listdir(base_dir) if f.endswith('.npz')])
        predictions = np.stack([
            np.load(os.path.join(base_dir, f))['predictions']
            for f in files
        ], axis=0)
        obs = np.load(os.path.join(base_dir, files[0]))['obs']

        print('[INFO] Single model:')
        if self.num_processes > 1:
            with Pool(self.num_processes) as pool:
                results = pool.map(self._eval_parallel, [(pred, obs) for pred in predictions])
        else:
            results = [
                self.eval_predictions(pred, obs)
                for pred in tqdm(predictions)
            ]
        results_single = pd.concat(results, axis=1).transpose()
        print(results_single.describe())

        if len(predictions) > ENSEMBLE_SIZE:
            print(f'[INFO] Ensemble ({ENSEMBLE_SIZE} members):')
            predictions_ens = [
                self.sample_ensemble_prediction(predictions)
                for _ in range(50)
            ]
            if self.num_processes > 1:
                with Pool(self.num_processes) as pool:
                    results = pool.map(self._eval_parallel, [(pred, obs) for pred in predictions_ens])
            else:
                results = [
                    self.eval_predictions(pred, obs)
                    for pred in predictions_ens
                ]
            results_ensemble = pd.concat(results, axis=1).transpose()
            print(results_ensemble.describe())
        else:
            results_ensemble = None

        print(f'[INFO] Total average ({len(predictions)} members):')
        results = self.eval_predictions(np.mean(predictions, axis=0), obs)
        results_complete = results.to_frame().transpose()

        print(results_complete.describe())


        return {
            'single': results_single,
            f'ensemble_{ENSEMBLE_SIZE}': results_ensemble,
            'complete': results_complete,
        }


    def store_results(self, results, run):
        output_dir = os.path.join(run.get_evaluation_path(), 'scores', self.mode)
        os.makedirs(output_dir, exist_ok=True)
        for key, vals in results.items():
            if vals is not None:
                vals.to_csv(os.path.join(output_dir, f'scores_{run.name}_{self.mode}_{key}.csv'))


    def eval_experiment(self, path: str):
        print(f'[INFO] Evaluating {path}')
        experiment = MultiRunExperiment(path, _except_on_not_existing=True)
        params = pd.DataFrame(experiment.list_run_parameters())
        params = params.drop([
            c for c in params.columns
            if (c not in {'run_name', 'time_stamp'}) and (np.all(params[c] == params[c].iloc[0]) or np.all(params[c].isnull()))
        ], axis=1)
        for i in range(len(params)):
            print('[INFO] Parameters:')
            print(params.iloc[i])
            run = experiment.load_run(params['run_name'].iloc[i])
            self.eval_run(run)

    def eval_run(self, run: PyTorchRun):
        npz_path = os.path.join(run.get_evaluation_path(), 'predictions', self.mode)
        print('[INFO] Results:')
        results = self.eval_npz(npz_path)
        self.store_results(results, run)

    def eval_or_recurse(self, path, max_depth=None):
        try:
            self.eval_experiment(path)
            success = True
        except AssertionError as e:
            print(e)
            success = False
        if not success:
            print('[INFO] Recursing...')
            if max_depth is None or max_depth > 0:
                contents = [c for c in os.listdir(path) if os.path.isdir(os.path.join(path, c))]
                new_max_depth = max_depth - 1 if max_depth is not None else None
                for c in sorted(contents):
                    self.eval_or_recurse(os.path.join(path, c), max_depth=new_max_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('model', type=str, choices=['bqn', 'logistic', 'normal'])
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--num-members', required=True, type=int)
    parser.add_argument('--num-processes', default=8, type=int)
    parser.add_argument('--valid', action='store_true', dest='eval_valid')
    parser.add_argument('--test', action='store_true', dest='eval_test')
    parser.add_argument('--forecasts', action='store_true', dest='eval_forecasts')
    parser.set_defaults(eval_valid=False, eval_test=False, eval_forecasts=False)
    args = vars(parser.parse_args())

    modes = []
    for mode in ['valid', 'test', 'forecasts']:
        if args[f'eval_{mode}']:
            modes.append(mode)

    if len(modes) == 0:
        print('[INFO] No data folds for evaluation. Exiting.')

    for mode in modes:
        eval = Eval(mode, args['model'], args['num_members'], num_processes=args['num_processes'])
        eval_mode = args['mode']
        if eval_mode == 'npz':
            eval.eval_npz(args['path'])
        elif eval_mode == 'run':
            _, run = MultiRunExperiment.from_run_path(args['path'], _except_on_not_existing=True, return_run=True)
            eval.eval_run(run)
        elif eval_mode == 'retrain':
            eval.eval_retrain(args['path'])
        elif eval_mode == 'exp':
            eval.eval_experiment(args['path'])
        elif eval_mode == 'r':
            eval.eval_or_recurse(args['path'])
        else:
            raise Exception(f'[ERROR] Unknown mode: {mode}')



if __name__ == '__main__':
    main()
