import argparse

import numpy as np
import torch
import tqdm
from torch import nn, Tensor

from data.euppbench.reforecasts import DataConfig
from experiments.helpers.optimizer import build_optimizer
from utils.progress import WelfordStatisticsTracker


class EnsembleMedianFeatures(nn.Module):

    def __init__(self, dim=-2):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return (x > torch.median(x, dim=self.dim, keepdim=True).values).to(dtype=torch.float32)


class Pretraining(object):

    def __init__(self, feature_module, num_epochs, prep_func):
        self.feature_module = feature_module
        self.num_epochs = num_epochs
        self.prepare_data = prep_func

    @classmethod
    def from_args(cls, args, prep_func, dim=-2):
        mode = args['pre_training:mode']
        if mode == 'median':
            features = EnsembleMedianFeatures(dim=dim)
        else:
            raise NotImplementedError()
        num_epochs = args['pre_training:num_epochs']
        return cls(features, num_epochs, prep_func)

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('pretraining')
        group.add_argument('--pre-training:num-epochs', type=int, default=0)
        group.add_argument('--pre-training:mode', type=str, default='median', choices=['median'])

    def run(self, args, model, training_loader, conditions, device):
        encoder = model['model'].encoder
        if encoder is None:
            raise Exception('[ERROR] Model does not possess an encoder. Encoder is needed for pre-training.')

        print('[INFO] Running pretraining')
        classifier = nn.Conv1d(encoder.out_channels, DataConfig.from_args(args).num_channels(), (1,)).to(device).train()
        pre_model = nn.ModuleDict({'model': model, 'decoder': classifier})
        optimizer = build_optimizer(args, pre_model)
        loss_function = nn.BCEWithLogitsLoss(reduction='sum')
        training_tracker = WelfordStatisticsTracker()
        for e in range(self.num_epochs):
            training_tracker.reset()
            model.train()
            with tqdm.tqdm(total=len(training_loader)) as pbar:
                for i, batch in enumerate(training_loader):
                    pre_model.zero_grad()
                    predictors, observations = self.prepare_data(batch, conditions, model, device, pretraining=self.feature_module)
                    features = model['model'].compute_member_codes(*predictors)
                    predictions = classifier(features)
                    loss = loss_function(predictions, observations)
                    loss.backward()
                    optimizer.step()
                    pbar.update()
                    num_samples = int(np.prod(observations.shape))
                    training_tracker.update(loss.item() / num_samples, weight=num_samples)
            print(f'[INFO] Pretraining Epoch {e + 1}: BCE (Training) = {training_tracker.mean()}')


