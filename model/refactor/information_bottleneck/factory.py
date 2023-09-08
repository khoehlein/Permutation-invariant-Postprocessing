import argparse

from torch import nn

from model.refactor.factory import (
    _build_station_embedding, _build_yday_embedding,
    _build_encoder, _build_merger, _build_decoder
)
from model.refactor.information_bottleneck.information_bottleneck import (
    InformationBottleneckV1,
    MultiFeatureBottleneck, InformationBottleneckV2
)
from model.refactor.processor import EnsembleProcessor


def init_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('model')
    bn_group = group.add_argument_group('info-bn')
    bn_group.add_argument('--model:bottleneck:type', type=str, default='v1', choices=['v1', 'v2'])
    bn_group.add_argument('--model:bottleneck:embedding-dim', type=int, default=8)
    bn_group.add_argument('--model:bottleneck:backbone-kws', type=str, default='{}')
    enc_group = group.add_argument_group('encoder')
    enc_group.add_argument('--model:encoder:type', type=str, default='mlp', choices=['mlp', 'resnet', 'attention', 'rescale'])
    enc_group.add_argument('--model:encoder:num-layers', type=int, default=2)
    enc_group.add_argument('--model:encoder:num-channels', type=int, default=64)
    enc_group.add_argument('--model:encoder:dropout', type=float, default=0.)
    enc_group.add_argument('--model:encoder:kwargs', type=str, default=None)
    dec_group = group.add_argument_group('decoder')
    dec_group.add_argument('--model:decoder:type', type=str, default='mlp', choices=['mlp', 'resnet', 'schulz-lerch'])
    dec_group.add_argument('--model:decoder:num-layers', type=int, default=2)
    dec_group.add_argument('--model:decoder:num-channels', type=int, default=64)
    dec_group.add_argument('--model:decoder:dropout', type=float, default=0.)
    dec_group.add_argument('--model:decoder:kwargs', type=str, default=None)
    mer_group = group.add_argument_group('merger')
    mer_group.add_argument('--model:merger:type', type=str, default='mean', choices=['mean', 'mean-std', 'mean-logstd', 'max', 'min-max', 'weighted-mean', 'weighted-mean-std', 'weighted-mean-logstd'])
    mer_group.add_argument('--model:merger:kwargs', type=str, default='{}')
    mer_group.add_argument('--model:merger:feature-selector:type', type=str, choices=['smallify', 'dropout'], default='smallify')
    mer_group.add_argument('--model:merger:feature-selector:weight', type=float, default=0.)
    mer_group.add_argument('--model:merger:feature-selector:kwargs', type=str, default='{}')
    con_group= group.add_argument_group('conditions')
    con_group.add_argument('--model:yday-embedding:type', type=str, default='cyclic', choices=['cyclic', 'linear', 'learned', 'cosine'])
    con_group.add_argument('--model:yday-embedding:kwargs', type=str, default='{}')
    con_group.add_argument('--model:station-embedding', type=int, default=10)
    group.add_argument('--model:bottleneck', type=int, default=None)
    group.set_defaults(add_input_conditioning=True, add_bottleneck_conditioning=True)


def _build_bottleneck(args, channels):
    bn_type = args['model:bottleneck:type']
    if bn_type == 'v1':
        return InformationBottleneckV1.from_args(args, channels)
    elif bn_type =='v2':
        return InformationBottleneckV2.from_args(args, channels)
    else:
        raise NotImplementedError(f'[ERROR] Unknown bottleneck type: {bn_type}')


def build_model(
        args,
        num_stations, condition_channels,
        in_channels, out_channels
):
    ensemble_bn = _build_bottleneck(args, in_channels)
    station_embedding = _build_station_embedding(args, num_stations)
    assert station_embedding is not None
    yday_embedding = _build_yday_embedding(args)
    assert yday_embedding is not None
    yday_bn_enc = _build_bottleneck(args, yday_embedding.out_channels())
    condition_bn_enc = _build_bottleneck(args, condition_channels)
    encoder_conditions = station_embedding.out_channels() + yday_bn_enc.out_channels + condition_bn_enc.out_channels
    encoder = _build_encoder(args, ensemble_bn.out_channels, encoder_conditions)
    bottleneck_channels = encoder.out_channels
    merger = _build_merger(args, bottleneck_channels)
    decoder_channels = merger.output_channels(bottleneck_channels)
    yday_bn_dec = _build_bottleneck(args, yday_embedding.out_channels())
    condition_bn_dec = _build_bottleneck(args, condition_channels)
    decoder_conditions = station_embedding.out_channels() + yday_bn_dec.out_channels + condition_bn_dec.out_channels
    decoder = _build_decoder(args, decoder_channels, decoder_conditions, out_channels)
    assert decoder is not None
    processor = EnsembleProcessor(merger=merger, encoder=encoder, decoder=decoder,)
    model = nn.ModuleDict({
        'station_embedding': station_embedding,
        'yday_embedding': yday_embedding,
        'model': processor,
        'bottleneck_encoder': MultiFeatureBottleneck({
            'yday': yday_bn_enc,
            'conditions': condition_bn_enc
        }),
        'bottleneck_decoder': MultiFeatureBottleneck({
            'yday': yday_bn_dec,
            'conditions': condition_bn_dec
        }),
        'bottleneck_ensemble': ensemble_bn,
    })
    return model
