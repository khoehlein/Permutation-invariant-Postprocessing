import os
import json

from ..helpers import get_timestamp_string


class BaseRun(object):

    TIMESTAMP_FORMAT = '%Y%m%d%H%M%S%f'
    DESCRIPTION_FILE_NAME = 'description.json'

    def __init__(self, path, description=None, _max_epoch=9999, _experiment=None):
        self.path = os.path.abspath(path)
        self._max_epoch = _max_epoch
        self.description = None
        if os.path.isdir(self.path):
            contents = os.listdir(self.path)
            if len(contents) > 0:
                assert description is None, ' '.join([
                    '[ERROR] {} is not an empty directory.',
                    'Descriptions may only be given for empty run directories.'.format(self.path)
                ])
                self._read_directory()
            else:
                self._setup_directory(description)
        else:
            os.makedirs(path)
            self._setup_directory(description)
        self.experiment = _experiment

    @property
    def name(self):
        return os.path.split(self.path)[-1]

    class DescriptionKey(object):
        TIMESTAMP = 'timestamp'
        PARAMETERS = 'parameters'
        MAX_EPOCH = 'max_epoch'
        RUN_TYPE = 'run_type'

    class Directory(object):
        SUMMARY = 'summary'
        EVALUATION = 'evaluation'

    def _blocked_keys(self):
        return {
            self.DescriptionKey.TIMESTAMP,
            self.DescriptionKey.MAX_EPOCH,
            self.DescriptionKey.PARAMETERS,
            self.DescriptionKey.RUN_TYPE
        }

    def _read_directory(self):
        assert os.path.isdir(self.get_summary_path())
        assert os.path.exists(self.get_description_file_path())
        self.description = self.read_description()
        assert self.DescriptionKey.MAX_EPOCH in self.description
        self._max_epoch = self.description[self.DescriptionKey.MAX_EPOCH]

    def _setup_directory(self, description):
        os.makedirs(self.get_summary_path())
        if description is not None:
            assert type(description) == dict
            occupied_keys = set(description.keys()).intersection(self._blocked_keys())
            assert len(occupied_keys) == 0, \
                '[ERROR] The following keys may not be used for description dicts: {}'.format(occupied_keys)
        else:
            description = {}
        self.description = description
        self.description.update({
            self.DescriptionKey.RUN_TYPE: self.__class__.__name__,
            self.DescriptionKey.TIMESTAMP: get_timestamp_string(self.TIMESTAMP_FORMAT),
            self.DescriptionKey.MAX_EPOCH: self._max_epoch,
            self.DescriptionKey.PARAMETERS: {}
        })
        self._dump_description()

    def add_parameter_settings(self, settings):
        assert type(settings) == dict
        description_params = self.description[self.DescriptionKey.PARAMETERS]
        overlap = set(description_params.keys()).intersection(set(settings.keys()))
        assert len(overlap) == 0, \
            '[ERROR] Description already contains settings for parameters {}'.format(list(overlap))
        description_params.update(settings)
        self._dump_description()

    def _dump_description(self):
        with open(self.get_description_file_path(), 'w') as description_file:
            json.dump(self.description, description_file, indent=4, sort_keys=True)

    def read_description(self):
        with open(self.get_description_file_path(), 'r') as f:
            description = json.load(f)
        return description

    def get_summary_path(self):
        return os.path.join(self.path, self.Directory.SUMMARY)

    def get_description_file_path(self):
        return os.path.join(self.path, self.DESCRIPTION_FILE_NAME)

    def get_evaluation_path(self):
        eval_path = os.path.join(self.path, self.Directory.EVALUATION)
        if not os.path.isdir(eval_path):
            os.makedirs(eval_path)
        return eval_path

    def parameters(self):
        return self.description[self.DescriptionKey.PARAMETERS]

    def is_empty(self):
        return True
