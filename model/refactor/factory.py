import argparse
import ast
from math import sqrt

from torch import nn

from model.dequant.uniform import UniformDequantization
from model.embeddings.multi_feature_embedding import MultiFeatureEmbedding
from model.embeddings.stations.builtin_embedding import BuiltinEmbedding
from model.refactor.ensemble.attention import EnsembleAttention
from model.feature_selection import SmallifySelector, VariationalDropoutSelector
from model.embeddings.yday.yday_features import CyclicYearDayEmbedding, LinearYearDayEmbedding, \
    LearnedYearDayEmbedding, CosineYearDayEmbedding
from model.refactor.processor import EnsembleProcessor, Identity
from model.ensemble.merger import MeanMerger, MeanStdMerger, MaxMerger, MinMaxMerger, SelectionWrapper, \
    WeightedMeanMerger, WeightedMeanStdMerger
from model.refactor.ensemble.mlp import MLP, SchulzLerchMLP
from model.refactor.ensemble.resnet import ResNet
from model.refactor.processor import Regressor, Rescaler
from model.scaling import StandardScaler


def init_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('model')
    enc_group = group.add_argument_group('encoder')
    enc_group.add_argument('--model:encoder:type', type=str, default='mlp', choices=['mlp', 'resnet', 'attention', 'rescale'])
    enc_group.add_argument('--model:encoder:num-layers', type=int, default=4)
    enc_group.add_argument('--model:encoder:num-channels', type=int, default=64)
    enc_group.add_argument('--model:encoder:dropout', type=float, default=0.)
    enc_group.add_argument('--model:encoder:kwargs', type=str, default=None)
    dec_group = group.add_argument_group('decoder')
    dec_group.add_argument('--model:decoder:type', type=str, default='mlp', choices=['mlp', 'resnet', 'schulz-lerch'])
    dec_group.add_argument('--model:decoder:num-layers', type=int, default=3)
    dec_group.add_argument('--model:decoder:num-channels', type=int, default=64)
    dec_group.add_argument('--model:decoder:dropout', type=float, default=0.)
    dec_group.add_argument('--model:decoder:kwargs', type=str, default=None)
    mer_group = group.add_argument_group('merger')
    mer_group.add_argument('--model:merger:type', type=str, default='weighted-mean', choices=['mean', 'mean-std', 'mean-logstd', 'max', 'min-max', 'weighted-mean', 'weighted-mean-std', 'weighted-mean-logstd'])
    mer_group.add_argument('--model:merger:kwargs', type=str, default='{}')
    mer_group.add_argument('--model:merger:feature-selector:type', type=str, choices=['smallify', 'dropout'], default='smallify')
    mer_group.add_argument('--model:merger:feature-selector:weight', type=float, default=0.)
    mer_group.add_argument('--model:merger:feature-selector:kwargs', type=str, default='{}')
    con_group= group.add_argument_group('conditions')
    con_group.add_argument('--model:yday-embedding:type', type=str, default=None, choices=['cyclic', 'linear', 'learned', 'cosine'])
    con_group.add_argument('--model:yday-embedding:kwargs', type=str, default='{}')
    con_group.add_argument('--model:station-embedding', type=int, default=10)
    con_group.add_argument('--model:add-input-conditioning', dest='add_input_conditioning', action='store_true')
    con_group.add_argument('--model:add-bottleneck-conditioning', dest='add_bottleneck_conditioning', action='store_true')
    deq_group = group.add_argument_group('dequantization')
    deq_group.add_argument('--model:dequant:type', type=str, default=None, choices=['uniform', 'none'])
    group.add_argument('--model:bottleneck', type=int, default=None)
    group.add_argument('--model:normalization', type=str, default=None, choices=['standardize'])
    group.add_argument('--model:feature-selector:type', type=str, choices=['smallify', 'dropout'], default='smallify')
    group.add_argument('--model:feature-selector:weight', type=float, default=0.)
    group.add_argument('--model:feature-selector:kwargs', type=str, default='{}')
    group.set_defaults(add_input_conditioning=False, add_bottleneck_conditioning=False)


def _build_station_embedding(args, num_stations):
    dim = args['model:station_embedding']
    if dim > 0:
        return BuiltinEmbedding(num_stations, dim)
    return None


def _build_yday_embedding(args):
    yday_type = args['model:yday_embedding:type']
    kwargs = args['model:yday_embedding:kwargs']
    if kwargs is None:
        kwargs = {}
    else:
        kwargs = ast.literal_eval(kwargs)
    if yday_type is None:
        return None
    elif yday_type == 'linear':
        yday_embedding = LinearYearDayEmbedding()
    elif yday_type == 'cyclic':
        yday_embedding = CyclicYearDayEmbedding()
    elif yday_type == 'cosine':
        yday_embedding = CosineYearDayEmbedding()
    elif yday_type == 'learned':
        yday_embedding = LearnedYearDayEmbedding(**kwargs)
    else:
        raise NotImplementedError()
    return yday_embedding


def _build_merger(args, bottleneck_channels: int):
    merger_type = args['model:merger:type']
    kwargs = args['model:merger:kwargs']
    if kwargs is None:
        kwargs = {}
    else:
        kwargs = ast.literal_eval(kwargs)
    if merger_type == 'mean':
        merger = MeanMerger()
    elif merger_type == 'mean-std':
        merger = MeanStdMerger(log_std=False, eps=1.e-6)
    elif merger_type == 'mean-logstd':
        merger = MeanStdMerger(log_std=True, eps=1.e-6)
    elif merger_type == 'max':
        merger = MaxMerger()
    elif merger_type == 'min-max':
        merger = MinMaxMerger()
    elif merger_type == 'weighted-mean':
        merger = WeightedMeanMerger(bottleneck_channels, **kwargs)
    elif merger_type == 'weighted-mean-std':
        merger = WeightedMeanStdMerger(bottleneck_channels, eps=1.e-6, log_std=False, **kwargs)
    elif merger_type == 'weighted-mean-logstd':
        merger = WeightedMeanStdMerger(bottleneck_channels, eps=1.e-6, log_std=True, **kwargs)
    else:
        raise NotImplementedError()
    try:
        selector_weight = args['model:merger:feature_selector:weight']
    except KeyError:
        selector_weight = 0.
    if selector_weight > 0.:
        selector = _build_feature_selector(merger.output_channels(bottleneck_channels), args, prefix='model:merger')
        merger = SelectionWrapper(merger, selector)
    return merger


def _build_encoder(args, feature_channels, condition_channels):
    num_layers = args['model:encoder:num_layers']
    bottleneck_channels = args['model:bottleneck']
    architecture = args['model:encoder:type']
    if architecture == 'rescale':
        return Rescaler(feature_channels, condition_channels, bottleneck_channels)
    if num_layers > 0:
        kwargs = args['model:encoder:kwargs']
        if kwargs is None:
            kwargs = {}
        else:
            kwargs = ast.literal_eval(kwargs)
        if architecture == 'mlp':
            backbone = MLP(
                args['model:encoder:num_channels'],
                num_layers=num_layers - 1, dropout=args['model:encoder:dropout'],
                **kwargs
            )
        elif architecture == 'resnet':
            backbone = ResNet(
                args['model:encoder:num_channels'],
                num_layers=num_layers, dropout=args['model:encoder:dropout'],
                **kwargs
            )
        elif architecture == 'attention':
            backbone = EnsembleAttention(
                args['model:encoder:num_channels'],
                num_layers=num_layers, dropout=args['model:encoder:dropout'],
                **kwargs
            )
        else:
            raise NotImplementedError()
        return Regressor(feature_channels, condition_channels, bottleneck_channels, backbone)
    return Identity(feature_channels, condition_channels, bottleneck_channels)


def _build_decoder(args, bottleneck_channels, condition_channels, out_channels):
    num_layers = args['model:decoder:num_layers']
    if num_layers > 0:
        architecture = args['model:decoder:type']
        kwargs = args['model:decoder:kwargs']
        if kwargs is None:
            kwargs = {}
        else:
            kwargs = ast.literal_eval(kwargs)
        if architecture == 'mlp':
            backbone = MLP(
                args['model:decoder:num_channels'],
                num_layers=num_layers - 1, dropout=args['model:decoder:dropout'],
                **kwargs
            )
        elif architecture == 'resnet':
            backbone = ResNet(
                args['model:decoder:num_channels'],
                num_layers=num_layers, dropout=args['model:decoder:dropout'],
                **kwargs
            )
        elif architecture == 'schulz-lerch':
            backbone = SchulzLerchMLP(args['loss:type'])
        else:
            raise NotImplementedError()
        return Regressor(bottleneck_channels, condition_channels, out_channels, backbone)
    return None


def _build_scaler(args):
    scaler_type = args['model:normalization']
    if scaler_type is None:
        return None
    elif scaler_type == 'standardize':
        return StandardScaler(channel_dim=-1)
    else:
        raise NotImplementedError()


def _init(m):

    def __init(m, fi, fo):
        b = sqrt(12. / (fi + fo))
        nn.init.uniform_(m.weight.data, -b, b)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

    if isinstance(m, nn.Linear):
        fi, fo = m.in_features, m.out_features
        __init(m, fi, fo)
    if isinstance(m, nn.Conv1d):
        fi, fo = m.in_channels, m.out_channels
        __init(m, fi, fo)


def _build_feature_selector(in_channels, args, prefix='model'):
    if args[prefix + ':feature_selector:weight'] <= 0:
        return None
    fs_type = args[prefix + ':feature_selector:type']
    kwargs = ast.literal_eval(args[prefix + ':feature_selector:kwargs'])
    if fs_type == 'smallify':
        return SmallifySelector(in_channels, **kwargs)
    elif fs_type == 'dropout':
        return VariationalDropoutSelector(in_channels, **kwargs)
    else:
        raise NotImplementedError()


def _build_dequantization(args):
    if 'model:dequant:type' in args:
        dequant_type = args['model:dequant:type']
        if dequant_type is None:
            return None
        elif dequant_type == 'uniform':
            return UniformDequantization()
        elif dequant_type == 'none':
            return None
        else:
            raise NotImplementedError(f'[ERROR] Unknown dequantizer type {dequant_type}')
    return None


def build_model(
        args,
        num_stations, condition_channels,
        in_channels, out_channels
):
    station_embedding = _build_station_embedding(args, num_stations)
    yday_embedding = _build_yday_embedding(args)
    encoder_conditions = 0
    if args['add_input_conditioning']:
        encoder_conditions += condition_channels
        if station_embedding is not None:
            encoder_conditions += station_embedding.out_channels()
        if yday_embedding is not None:
            encoder_conditions += yday_embedding.out_channels()
    encoder = _build_encoder(args, in_channels, encoder_conditions)
    bottleneck_channels = encoder.out_channels
    merger = _build_merger(args, bottleneck_channels)
    decoder_channels = merger.output_channels(bottleneck_channels)
    decoder_conditions = 0
    if args['add_bottleneck_conditioning']:
        decoder_conditions += condition_channels
        if station_embedding is not None:
            decoder_conditions += station_embedding.out_channels()
        if yday_embedding is not None:
            decoder_conditions += yday_embedding.out_channels()
    decoder = _build_decoder(args, decoder_channels, decoder_conditions, out_channels)
    if decoder is None:
        merger_channels = merger.output_channels(bottleneck_channels)
        assert merger_channels == out_channels, \
            f'[ERROR] Expected {out_channels} output channels, but found {merger_channels} channels after merger'
    scaler = _build_scaler(args)
    processor = EnsembleProcessor(merger=merger, encoder=encoder, decoder=decoder,)
    # processor.apply(_init)
    feature_selector = _build_feature_selector(in_channels, args, prefix='model')
    dequant = _build_dequantization(args)
    multi_embedding = MultiFeatureEmbedding({
        'yday': yday_embedding,
        'locations': station_embedding
    })
    model = nn.ModuleDict({
        'station_embedding': station_embedding,
        'yday_embedding': yday_embedding,
        'embeddings': multi_embedding,
        'model': processor,
        'scaler': scaler,
        'feature_selector': feature_selector,
        'dequant': dequant,
    })
    return model
