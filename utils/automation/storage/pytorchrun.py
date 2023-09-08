import os
import torch
from torch.utils.tensorboard import SummaryWriter

from .baserun import BaseRun
from ..helpers import get_timestamp_string


class PyTorchRun(BaseRun):

    CHECKPOINT_EXTENSION = '.pth'

    def _read_directory(self):
        super(PyTorchRun, self)._read_directory()
        assert os.path.isdir(self.get_checkpoint_path())

    def _setup_directory(self, description):
        super(PyTorchRun, self)._setup_directory(description)
        os.makedirs(self.get_checkpoint_path())

    class Directory(BaseRun.Directory):
        CHECKPOINTS = 'checkpoints'

    def get_checkpoint_path(self):
        return os.path.join(self.path, self.Directory.CHECKPOINTS)

    def list_checkpoints(self, sort_output=True):
        checkpoints = [f for f in os.listdir(self.get_checkpoint_path()) if f.endswith(self.CHECKPOINT_EXTENSION)]
        return sorted(checkpoints) if sort_output else list(checkpoints)

    def save_checkpoint(self, checkpoint_state, file_name):
        if not file_name.endswith(self.CHECKPOINT_EXTENSION):
            file_name = file_name + self.CHECKPOINT_EXTENSION
        checkpoint_path = os.path.join(self.get_checkpoint_path(), file_name)
        torch.save(checkpoint_state, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, file_name, map_location=None):
        if not file_name.endswith(self.CHECKPOINT_EXTENSION):
            file_name = file_name + self.CHECKPOINT_EXTENSION
        checkpoint_path = os.path.join(self.get_checkpoint_path(), file_name)
        return torch.load(checkpoint_path, map_location=map_location)

    def _get_epoch_file_name(self, epoch):
        assert epoch <= self._max_epoch, \
            '[ERROR] Epoch number {} is larger than max. expected epoch number {}.'.format(epoch, self._max_epoch)
        max_digits = len(str(self._max_epoch))
        file_name_format = 'epoch_{:0' + '{}'.format(max_digits) + 'd}' + self.CHECKPOINT_EXTENSION
        file_name = file_name_format.format(epoch)
        return file_name

    def save_epoch_state(self, epoch_state, epoch):
        file_name = self._get_epoch_file_name(epoch)
        return self.save_checkpoint(epoch_state, file_name)

    def load_epoch_state(self, epoch, map_location=None):
        if epoch >= 0:
            file_name = self._get_epoch_file_name(epoch)
        else:
            all_epochs = [f for f in self.list_checkpoints(sort_output=True) if f.startswith('epoch_')]
            file_name = all_epochs[epoch]
        return self.load_checkpoint(file_name, map_location=map_location)

    def is_empty(self):
        return len(self.list_checkpoints()) == 0 and len(os.listdir(self.get_summary_path())) == 0

    def get_tensorboard_summary(self):
        timestamp = get_timestamp_string()
        file_path = os.path.join(self.get_summary_path(), timestamp)
        os.makedirs(file_path)
        return SummaryWriter(file_path)
