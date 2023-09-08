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


def get_paths(config_name: str, assert_not_none = True) -> str:
    host_name = socket.gethostname()
    paths = read_path_configs(config_name).get(host_name, None)
    if assert_not_none:
        assert paths is not None, f'[ERROR] Hostname {host_name} not found in configuration {config_name}'
    for key in paths:
        data = paths[key]
        data = {int(key): path for key, path in data.items()}
        paths[key] = data
    return paths
