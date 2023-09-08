import os
import time
import subprocess
from itertools import chain


class ComputeJob(object):
    def __init__(
            self,
            executable_path, script_path, cwd=None,
            posargs=None, kwargs=None, flags=None,
            id=None,
            _verify_script=True, _verify_executable=True, _verify_cwd=True
    ):
        self.id = id
        if _verify_executable:
            assert os.path.exists(executable_path)
        self.executable_path = executable_path
        if _verify_script:
            assert os.path.exists(script_path)
        self.script_path = script_path
        if cwd is not None and _verify_cwd:
            assert os.path.isdir(cwd)
        self.cwd = cwd
        if posargs is not None:
            assert type(posargs) == list
        self.posargs = posargs
        if kwargs is not None:
            assert type(kwargs) == dict
        self.kwargs = kwargs
        if flags is not None:
            assert type(flags) == list
        self.flags = flags

    @classmethod
    def from_dict(cls, config, id=None, _verify_script=True, _verify_executable=True):
        executable_path = config[cls.ConfigKey.EXECUTABLE]
        script_path = config[cls.ConfigKey.SCRIPT]
        cwd = config[cls.ConfigKey.CWD] if ComputeJob.ConfigKey.CWD in config else None
        posargs = config[cls.ConfigKey.POSARGS] if cls.ConfigKey.POSARGS in config else None
        kwargs = config[cls.ConfigKey.KWARGS] if cls.ConfigKey.KWARGS in config else None
        flags = config[cls.ConfigKey.FLAGS] if cls.ConfigKey.FLAGS in config else None
        return cls(
            executable_path, script_path, cwd=cwd,
            posargs=posargs, kwargs=kwargs, flags=flags, id=id,
            _verify_script=_verify_script, _verify_executable=_verify_executable
        )

    def to_dict(self):
        config = {
            ComputeJob.ConfigKey.EXECUTABLE: self.executable_path,
            ComputeJob.ConfigKey.SCRIPT: self.script_path,
            ComputeJob.ConfigKey.CWD: self.cwd,
            ComputeJob.ConfigKey.POSARGS: self.posargs,
            ComputeJob.ConfigKey.KWARGS: self.kwargs,
            ComputeJob.ConfigKey.FLAGS: self.flags
        }
        return config

    def execution_command(self):
        posargs_string = ['{}'.format(arg) for arg in self.posargs]
        kwargs_string = ['{}'.format(arg) for arg in chain.from_iterable(self.kwargs.items())]
        flags_string = ['{}'.format(arg) for arg in self.flags]
        return [self.executable_path, self.script_path] + posargs_string + kwargs_string + flags_string

    def run(self, log_file=None):
        if log_file is not None:
            log_file = open(log_file, 'w')
            stdout = log_file
            stderr = log_file
        else:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        time_start = time.time()
        process = subprocess.Popen(self.execution_command(), cwd=self.cwd, stdout=stdout, stderr=stderr)
        time_end = time.time()
        outputs, errors = process.communicate()
        if log_file is not None:
            log_file.close()
        duration = (time_end - time_start)
        log = {
            ComputeJob.LogKey.JOB_ID: self.id,
            ComputeJob.LogKey.PROCESS_ID: process.pid,
            ComputeJob.LogKey.RETURNCODE: process.returncode,
            ComputeJob.LogKey.OUTPUTS: None if outputs is None else outputs.decode(),
            ComputeJob.LogKey.ERRORS: None if errors is None else errors.decode(),
            ComputeJob.LogKey.STARTTIME: time_start,
            ComputeJob.LogKey.ENDTIME: time_end,
            ComputeJob.LogKey.DURATION: duration,
        }
        return log

    class ConfigKey(object):
        EXECUTABLE = 'executable'
        SCRIPT = 'script'
        CWD = 'cwd'
        POSARGS = 'posargs'
        KWARGS = 'kwargs'
        FLAGS = 'flags'

    class LogKey(object):
        JOB_ID = 'job_id'
        PROCESS_ID = 'process_id'
        RETURNCODE = 'returncode'
        OUTPUTS = 'outputs'
        ERRORS = 'errors'
        STARTTIME = 'start'
        ENDTIME = 'end'
        DURATION = 'duration'