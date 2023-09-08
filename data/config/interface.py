import functools
import json
import os
import socket


@functools.lru_cache(maxsize=3)
def read_path_configs(file_name: str):
    config_dir = os.path.dirname(os.path.abspath(__file__))
    if not file_name.endswith('.json'):
        file_name = file_name + '.json'
    file_path = os.path.join(config_dir, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_path(config_name: str, assert_not_none = True) -> str:
    host_name = socket.gethostname()
    path = read_path_configs(config_name).get(host_name, None)
    if assert_not_none:
        assert path is not None, f'[ERROR] Hostname {host_name} not found in configuration {config_name}'
    return path


def get_project_base_path():
    return get_path('project_base')


def get_results_base_path():
    return get_path('project_base')


def get_interpreter_path():
    return get_path('interpreter')


def _test():
    example = read_path_configs('example')
    print(example)
    print(example['hostname'])


if __name__ == '__main__':
    _test()
