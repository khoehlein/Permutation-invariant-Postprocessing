import argparse
import os
import shutil

from utils.automation.storage import MultiRunExperiment


def get_log_path(run):
    log_base_path = os.path.join(run.get_evaluation_path(), 'log')
    contents = [f for f in os.listdir(log_base_path) if os.path.isdir(os.path.join(log_base_path, f))]
    assert len(contents) == 1
    return os.path.join(log_base_path, contents[0])


def get_target_path(run, mode):
    return os.path.join(run.get_evaluation_path(), 'predictions', mode)


def copy_run_data(run):
    log_path = get_log_path(run)
    files = list(sorted([f for f in os.listdir(log_path) if f.endswith('.npz')]))
    for mode in ['test', 'valid']:
        target_path = get_target_path(run, mode)
        os.makedirs(target_path, exist_ok=True)
        for f in files:
            if mode in f:
                shutil.copy(os.path.join(log_path, f), os.path.join(target_path, f))
    return files


def copy_or_recurse(path):
    success = False
    try:
        experiment = MultiRunExperiment(path, _except_on_not_existing=True)
        success = True
    except AssertionError:
        pass
    else:
        print(f'[INFO] Processing experiment {path}')
        experiment.map_across_runs(copy_run_data)
    if not success:
        print('[INFO] Recursing...')
        contents = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        for c in contents:
            copy_or_recurse(os.path.join(path, c))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = vars(parser.parse_args())
    copy_or_recurse(os.path.abspath(args['path']))


if __name__ == '__main__':
    main()
