import os
import random
from itertools import product
from .computejob import ComputeJob


class JobMaker(object):
    def __init__(
            self,
            interpreter_path, script_path, cwd=None,
            posargs=None, kwargs=None, flags=None,
            randomize=False, max_num_files=None,
            _verify_interpreter=True, _verify_script=True, _verify_cwd=True
    ):
        self.interpreter_path, self.script_path, self.cwd, self.posargs, self.kwargs, self.flags = self._parse_inputs(
            interpreter_path, script_path, cwd,
            posargs, kwargs, flags,
            _verify_interpreter, _verify_script, _verify_cwd
        )
        kwargs_list = [[(key, v) for v in values] for key, values in self.kwargs.items()]
        all_settings = list(product(*(self.posargs + kwargs_list + self.flags)))
        if randomize:
            all_settings = self._randomize(all_settings, max_num_files)
        elif max_num_files is None:
            pass
        elif max_num_files > 0:
            all_settings = all_settings[:min(max_num_files, len(all_settings))]
        else:
            raise Exception()
        num_posargs = len(self.posargs)
        num_kwargs = len(self.kwargs)
        self.job_dicts = [
            {
                ComputeJob.ConfigKey.EXECUTABLE: self.interpreter_path,
                ComputeJob.ConfigKey.SCRIPT: self.script_path,
                ComputeJob.ConfigKey.CWD: self.cwd,
                ComputeJob.ConfigKey.POSARGS: self._process_settings_list(setting[:num_posargs]),
                ComputeJob.ConfigKey.KWARGS: dict(setting[num_posargs:(num_posargs + num_kwargs)]),
                ComputeJob.ConfigKey.FLAGS: self._process_settings_list(setting[(num_posargs + num_kwargs):]),
            }
            for setting in all_settings
        ]

    @staticmethod
    def _process_settings_list(settings):
        out = []
        for setting in settings:
            if type(setting) == list:
                out += setting
            else:
                out.append(setting)
        return out

    @staticmethod
    def _parse_inputs(
            interpreter_path, script_path, cwd,
            posargs, kwargs, flags,
            _verify_interpreter, _verify_script, _verify_cwd
    ):
        if _verify_interpreter:
            assert os.path.exists(interpreter_path)
        interpreter_path = os.path.abspath(interpreter_path)
        if _verify_script:
            assert os.path.exists(script_path) and script_path.endswith('.py')
        script_path = os.path.abspath(script_path)
        if _verify_cwd and cwd is not None:
            assert os.path.isdir(cwd)
        if posargs is None:
            posargs = []
        assert type(posargs) == list
        posargs_parsed = [args if type(args) == list else [args] for args in posargs]
        if kwargs is None:
            kwargs = dict()
        assert type(kwargs) == dict
        kwargs_parsed = {key: args if type(args) == list else [args] for key, args in kwargs.items()}
        if flags is None:
            flags = []
        assert type(flags) == list
        flags_parsed = [flag if type(flag) == list else [flag] for flag in flags]
        return interpreter_path, script_path, cwd, posargs_parsed, kwargs_parsed, flags_parsed

    @staticmethod
    def _randomize(all_settings, max_num_files):
        if max_num_files is None:
            random.shuffle(all_settings)
        elif max_num_files > 0:
            all_settings = random.sample(all_settings, max_num_files)
        else:
            raise Exception()
        return all_settings

    def export_jobs(self, queue):
        jobs = [ComputeJob.from_dict(config) for config in self.job_dicts]
        job_ids = queue.add_jobs(*jobs)
        return job_ids
