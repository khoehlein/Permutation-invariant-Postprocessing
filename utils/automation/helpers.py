import os
from datetime import datetime


def get_run_directory(base_path):
    run_name = 'run_{}'.format(datetime.utcnow().strftime('%Y%m%d%H%M%S%f'))
    run_directory = os.path.join(base_path, run_name)
    summary_directory = os.path.join(run_directory, 'summary')
    checkpoint_directory = os.path.join(run_directory, 'checkpoints')
    os.makedirs(run_directory)
    os.mkdir(summary_directory)
    os.mkdir(checkpoint_directory)
    return run_name, summary_directory, checkpoint_directory


def get_timestamp_string(format=None):
    if format is None:
        format = '%Y%m%d%H%M%S%f'
    return '{}'.format(datetime.utcnow().strftime(format))
