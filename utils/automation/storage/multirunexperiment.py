import os
import json
from typing import Callable, Any

from ..helpers import get_timestamp_string

from .baserun import BaseRun
from .pytorchrun import PyTorchRun


class MultiRunExperiment(object):

    TIMESTAMP_FORMAT = '%Y%m%d%H%M%S%f'
    DESCRIPTION_FILE_NAME = 'description.json'

    @classmethod
    def from_run_path(cls, path, description=None, _except_on_not_existing=False, return_run=False):
        base, run_name = os.path.split(path)
        experiment = cls(os.path.abspath(os.path.join(base, '..')), description=description, _except_on_not_existing=_except_on_not_existing)
        run = experiment.load_run(run_name)
        if return_run:
            return experiment, run
        return experiment

    def __init__(self, path, description=None, _except_on_not_existing=False):
        self.path = os.path.abspath(path)
        if os.path.isdir(path):
            contents = set(os.listdir(self.path))
            if len(contents) > 0:
                assert description is None, ' '.join([
                    '[ERROR] {} is not an empty directory.',
                    'Descriptions may only be given for empty experiment directories.'.format(self.path)
                ])
                self._read_directory()
            else:
                self._setup_directory(description)
        else:
            if _except_on_not_existing:
                raise Exception(f'[ERROR] Directory {self.path} could not be found')
            os.makedirs(self.path)
            self._setup_directory(description)

    class DescriptionKey(object):
        TIMESTAMP = 'timestamp'

    class Directory(object):
        RUNS = 'runs'

    def _read_directory(self):
        assert os.path.isdir(self.get_run_directory_path()), '[ERROR] {} is not a valid experiment directory.'.format(
            self.path)
        assert os.path.exists(
            self.get_description_file_path()), '[ERROR] Experiment directory must contain a description file.'

    def _setup_directory(self, description):
        os.makedirs(self.get_run_directory_path())
        if description is not None:
            assert type(description) == dict
            assert self.DescriptionKey.TIMESTAMP not in description.keys()
        else:
            description = {}
        timestamp = get_timestamp_string(self.TIMESTAMP_FORMAT)
        description.update({self.DescriptionKey.TIMESTAMP: timestamp})
        with open(self.get_description_file_path(), 'w') as description_file:
            json.dump(description, description_file, indent=4, sort_keys=True)

    def get_run_directory_path(self):
        return os.path.join(self.path, self.Directory.RUNS)

    def get_description_file_path(self):
        return os.path.join(self.path, self.DESCRIPTION_FILE_NAME)

    def read_description(self):
        descrition_file_path = self.get_description_file_path()
        if os.path.exists(descrition_file_path):
            with open(descrition_file_path, 'r') as description_file:
                description = json.load(description_file)
        else:
            description = {}
        return description

    def _get_path_for_run(self, run_name):
        return os.path.join(self.path, self.Directory.RUNS, run_name)

    def create_new_run(self, run_name=None, description=None):
        if run_name is None:
            # use timestamp as run name
            while run_name is None or os.path.isdir(self._get_path_for_run(run_name)):
                run_name = get_timestamp_string(self.TIMESTAMP_FORMAT)
        path = self._get_path_for_run(run_name)
        return PyTorchRun(path, description=description, _experiment=self)

    def load_run(self, run_name: str) -> BaseRun:
        path = self._get_path_for_run(run_name)
        if os.path.isdir(path):
            base_run = BaseRun(path)
            description = base_run.description
            if base_run.DescriptionKey.RUN_TYPE in description:
                run_type = description[base_run.DescriptionKey.RUN_TYPE]
            else:
                run_type = 'PyTorchRun'
            if run_type == 'PyTorchRun':
                return PyTorchRun(path, _experiment=self)
            else:
                raise Exception('[ERROR] Run type <{}> could not be recognized'.format(run_type))
        else:
            raise Exception('[ERROR] Experiment does not contain a run with name <{}>.'.format(run_name))

    def list_runs(self, sort_output=True):
        directories = os.listdir(self.get_run_directory_path())
        return sorted(directories) if sort_output else list(directories)

    def list_run_parameters(self):
        data = {}
        run_names = self.list_runs(sort_output=True)
        timestamps = []
        for i, run_name in enumerate(run_names):
            run = self.load_run(run_name)
            timestamps.append(run.description[run.DescriptionKey.TIMESTAMP])
            params = run.parameters()
            param_keys = set(params.keys())
            data_keys = set(data.keys())
            known_keys = data_keys.intersection(param_keys)
            new_keys = param_keys - known_keys
            if len(new_keys):
                data.update({
                    key: ([None] * i) + [params[key]]
                    for key in new_keys
                })
            for key in known_keys:
                data[key].append(params[key])
            for key in (data_keys - known_keys):
                data[key].append(None)
        data.update({'run_name': run_names})
        data.update({'timestamp': timestamps})
        return data

    def map_across_runs(self, func: Callable[[BaseRun], Any], run_names=None, verbose=False):
        if run_names is None:
            run_names = self.list_runs()

        def eval(i: int, run_name: str):
            if verbose:
                print(f'[INFO] Processing run {run_name} ({i + 1}/{len(run_names)})')
            run = self.load_run(run_name)
            return func(run)

        return [eval(i, run_name) for i, run_name in enumerate(run_names)]
