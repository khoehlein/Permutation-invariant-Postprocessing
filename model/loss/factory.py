import argparse
import ast

from model.loss.losses import (
    LogisticCRPS, LogisticLogScore,
    NormalCRPS, NormalLogScore,
    LogNormalCRPS, LogNormalLogScore,
    BernsteinCRPS,
    BernsteinTransformedLogisticCRPS, BernsteinTransformedExponentialCRPS
)


def init_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('loss')
    group.add_argument('--loss:type', type=str, default='logistic', choices=['btexp', 'logistic', 'logistic-log', 'log-normal', 'bqn', 'btlq', 'normal'])
    group.add_argument('--loss:positive-constraint', type=str, default='softplus', choices=['softplus', 'exp'])
    group.add_argument('--loss:kwargs', type=str, default=None)


def build_loss(args, reduction='sum', for_eval=False):
    loss_type = args['loss:type']
    positive_constraint = args['loss:positive_constraint']
    try:
        kwargs = args['loss:kwargs']
    except KeyError:
        kwargs = None
    if kwargs is None:
        kwargs = '{}'
    kwargs = ast.literal_eval(kwargs)
    if reduction is None:
        reduction = 'none'
    if loss_type == 'logistic':
        loss = LogisticCRPS(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    elif loss_type == 'logistic-log':
        loss = LogisticLogScore(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    elif loss_type == 'log-normal':
        loss = LogNormalCRPS(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    elif loss_type == 'log-normal-log':
        loss = LogNormalLogScore(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    elif loss_type == 'normal':
        loss = NormalCRPS(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    elif loss_type == 'normal':
        loss = NormalLogScore(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    elif loss_type == 'bqn':
        if for_eval:
            kwargs.update({'integration_scheme': 'midpoint', 'num_quantiles': 200})
        loss = BernsteinCRPS(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    elif loss_type == 'btlq':
        if for_eval:
            kwargs.update({'integration_scheme': 'midpoint', 'num_quantiles': 200})
        loss = BernsteinTransformedLogisticCRPS(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    elif loss_type == 'btexp':
        if for_eval:
            kwargs.update({'integration_scheme': 'midpoint', 'num_quantiles': 200})
        loss = BernsteinTransformedExponentialCRPS(reduction=reduction, positive_constraint=positive_constraint, **kwargs)
    else:
        raise NotImplementedError()
    return loss
