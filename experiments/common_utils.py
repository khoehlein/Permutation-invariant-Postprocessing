import numpy as np
import torch
from torch.utils.data import Subset

from model.helpers import to_device_if_not


def split_for_validation(dataset, validation_share):
    split = np.floor(len(dataset) * (1. - validation_share))
    idx_train = torch.arange(split, dtype=torch.long)
    dataset_train = Subset(dataset, idx_train)
    idx_validation = torch.arange(split, len(dataset), dtype=torch.long)
    dataset_validation = Subset(dataset, idx_validation)
    return dataset_train, dataset_validation


def prepare_data(batch, conditions, model, device, pretraining=None, return_locations=False, use_dequant=False):
    features, yday, locations, observations = batch
    features = to_device_if_not(features, device)
    yday = to_device_if_not(yday, device)
    observations = to_device_if_not(observations, device)
    if pretraining is not None:
        observations = pretraining(features)
    else:
        if use_dequant and 'dequant' in model and model['dequant'] is not None:
            observations = model['dequant'](observations)
    # feature dimensions: batch, ensemble, channel
    features = features.detach()
    if model['feature_selector'] is not None:
        features = model['feature_selector'](features)
    supplements = [conditions[locations, :].detach()]
    if model['embeddings'] is not None:
        supplements.append(model['embeddings'](yday=yday, locations=locations))
    supplements = torch.cat(supplements, dim=-1)
    outputs = [(features, supplements), observations]
    if return_locations:
        outputs.append(locations)
    outputs = tuple(outputs)
    return outputs