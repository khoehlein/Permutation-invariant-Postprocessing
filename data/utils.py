from typing import Union

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset


class BatchLoader(object):

    def __init__(self, dataset: Union[Subset, TensorDataset], batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.indices = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        num_samples = len(self.dataset)
        if self.shuffle:
            indices = torch.randperm(num_samples, dtype=torch.long)
        else:
            indices = torch.arange(num_samples, dtype=torch.long)
        if self.drop_last:
            indices = indices[:(self.batch_size * (len(self.dataset) // self.batch_size))]
        indices = torch.chunk(indices, len(self))
        for i in indices:
            yield self.dataset[i]
