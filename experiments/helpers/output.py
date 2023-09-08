import argparse

from utils.automation.storage import MultiRunExperiment


def init_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('output')
    group.add_argument('--output:path', type=str, required=True)
    group.add_argument('--output:path:mode', type=str, choices=['run','exp'], default='exp')


def build_storage(args):
    path_mode = args.get('output:path:mode', None)
    if path_mode is None or path_mode == 'exp':
        experiment = MultiRunExperiment(args['output:path'])
        run = experiment.create_new_run()
    elif path_mode == 'run':
        experiment, run = MultiRunExperiment.from_run_path(args['output:path'], return_run=True)
    else:
        raise NotImplementedError(f'[ERROR] Unknown output path mode {path_mode}')
    run.add_parameter_settings(args)
    return experiment, run
