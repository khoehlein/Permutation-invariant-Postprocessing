import os
from itertools import chain

from data.config.interface import get_interpreter_path
from evaluation.prediction_directory import SELECTED_RUNS_EUPP
from utils.automation.queue import QueueManager, JobMaker


def main():
    experiment_path = os.path.dirname(os.path.abspath(__file__))
    queue_path = os.path.join(experiment_path, 'queue')
    queue = QueueManager(queue_path, make_directories=True)
    script_path = os.path.join(experiment_path, 'predict_perturbed.py')
    interpreter_path = get_interpreter_path()
    project_root_path = os.path.abspath(os.path.join(experiment_path, '../..', '..', '..'))
    run_paths = list(chain.from_iterable([x.values() for x in SELECTED_RUNS_EUPP.values()]))

    ensemble_shuffle = JobMaker(
        interpreter_path, script_path, cwd=project_root_path,
        posargs=[],
        kwargs={'--path': run_paths, '--seed': [42]},
        flags=['--gpu'],
    )
    ensemble_shuffle.export_jobs(queue)

    loc_shuffle = JobMaker(
        interpreter_path, script_path, cwd=project_root_path,
        posargs=[],
        kwargs={'--path': run_paths, '--loc-bins': [50, 100], '--scale-bins': [1], '--seed': [42]},
        flags=['--rerank', '--gpu']
    )
    loc_shuffle.export_jobs(queue)

    scale_shuffle = JobMaker(
        interpreter_path, script_path, cwd=project_root_path,
        posargs=[],
        kwargs={'--path': run_paths, '--loc-bins': [1], '--scale-bins': [50, 100], '--seed': [42]},
        flags=['--rerank', '--gpu']
    )
    scale_shuffle.export_jobs(queue)

    random_shuffle_ranked = JobMaker(
        interpreter_path, script_path, cwd=project_root_path,
        posargs=[],
        kwargs={'--path': run_paths, '--loc-bins': [1], '--scale-bins': [1], '--seed': [42]},
        flags=['--rerank', '--gpu']
    )
    random_shuffle_ranked.export_jobs(queue)

    random_shuffle = JobMaker(
        interpreter_path, script_path, cwd=project_root_path,
        posargs=[],
        kwargs={'--path': run_paths, '--loc-bins': [1], '--scale-bins': [1], '--seed': [42]},
        flags=['--gpu']
    )
    random_shuffle.export_jobs(queue)

    print(queue_path)


def main_v2():
    experiment_path = os.path.dirname(os.path.abspath(__file__))
    queue_path = os.path.join(experiment_path, 'queue')
    queue = QueueManager(queue_path, make_directories=True)
    script_path = os.path.join(experiment_path, 'predict_perturbed.py')
    interpreter_path = get_interpreter_path()
    project_root_path = os.path.abspath(os.path.join(experiment_path, '../..', '..', '..'))
    run_paths = list(chain.from_iterable([x.values() for x in SELECTED_RUNS_EUPP.values()]))

    metric_shuffle = JobMaker(
        interpreter_path, script_path, cwd=project_root_path,
        posargs=[],
        kwargs={'--path': run_paths, '--num-bins': [100], '--seed': [42]},
        flags=['--gpu']
    )
    metric_shuffle.export_jobs(queue)

    print(queue_path)


if __name__ == '__main__':
    main_v2()
