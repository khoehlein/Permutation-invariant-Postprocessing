import argparse
import ast

import torch


def _build_optimizer(args, model):
    if args['training:optimizer:kwargs'] is not None:
        kwargs = ast.literal_eval(args['training:optimizer:kwargs'])
    else:
        kwargs = {}
    return getattr(torch.optim, args['training:optimizer:type'])(
        model.parameters(), lr=args['training:optimizer:lr'], **kwargs
    )

def build_optimizer(args, model):
    optimizer = _build_optimizer(args, model)
    return optimizer


def init_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('optimizer')
    group.add_argument('--training:optimizer:type', type=str, default='Adam')
    group.add_argument('--training:optimizer:lr', type=float, default=5.e-4)
    group.add_argument('--training:optimizer:kwargs', type=str, default='{"weight_decay": 1.e-5}')