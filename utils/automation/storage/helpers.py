from typing import Callable, Any
import argparse

from .baserun import BaseRun
from .multirunexperiment import MultiRunExperiment


def map_across_experiment(func: Callable[[BaseRun], Any], verbose=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-path', type=str, required=True)
    parser.add_argument('--run-id', type=str, default=None)
    args = vars(parser.parse_args())
    experiment = MultiRunExperiment(args['experiment_path'])
    run_names = None
    if args['run_id'] is not None:
        run_names = [args['run_id'], ]
    return experiment.map_across_runs(func, run_names=run_names, verbose=verbose)
