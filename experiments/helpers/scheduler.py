import argparse
import ast

import torch


def init_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('scheduler')
    group.add_argument('--scheduler:type', type=str, default=None)
    group.add_argument('--scheduler:kwargs', type=str, default='{}')


def build_scheduler(args, optimizer):
    sched_type = args['scheduler:type']
    if sched_type is None:
        return None
    kwargs = ast.literal_eval(args['scheduler:kwargs'])
    try:
        sched_class = getattr(torch.optim.lr_scheduler, sched_type)
    except AttributeError:
        raise Exception(f'[ERROR] Unknown scheduler type: {sched_type}')
    else:
        return sched_class(optimizer, **kwargs)
