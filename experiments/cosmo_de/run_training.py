import argparse
import json

import numpy as np
import torch

from data.utils import BatchLoader
import torch.optim

from experiments.common_utils import prepare_data
from data.cosmo_de import DataConfig, build_training_dataset
from experiments.helpers.timer import Timer
from experiments.helpers.training import Trainer
from experiments.helpers import optimizer as optim_factory, output as output_factory, scheduler as sched_factory
from model.refactor import factory as model_factory
from model.loss import factory as loss_factory
from model.pretraining import Pretraining


# EXPERIMENT_BASE_PATH = '/path/to/somewhere'
# EMBEDDING_DIMENSION = 10
# HIDDEN_CHANNELS = 64
# BOTTLENECK_CHANNELS = None
# CONDITIONS_IN_BOTTLENECK = True
# ENSEMBLE_SIZE = 10
# VALIDATION_SHARE = 0.2
# LEARNING_RATE = 5.e-4
# NUM_EPOCHS = 150
# BATCH_SIZE = 64
# PATIENCE = 10

torch.set_num_threads(4)


def build_parser():
    parser = argparse.ArgumentParser()
    DataConfig.init_parser(parser)
    output_factory.init_parser(parser)
    group = parser.add_argument_group('training')
    group.add_argument('--training:batch-size', type=int, default=128)
    group.add_argument('--training:num-epochs', type=int, default=150)
    group.add_argument('--training:patience', type=int, default=10)
    group.add_argument('--training:ensemble-size', type=int, default=1)
    group.add_argument('--training:use-gpu', action='store_true', dest='use_gpu')
    group.add_argument('--training:use-cpu', action='store_false', dest='use_gpu')
    Pretraining.init_parser(parser)
    group.set_defaults(use_gpu=True)
    optim_factory.init_parser(parser)
    sched_factory.init_parser(parser)
    model_factory.init_parser(parser)
    loss_factory.init_parser(parser)
    return parser


def main():
    parser = build_parser()
    args = vars(parser.parse_args())

    device = torch.device('cuda:0' if args['use_gpu'] and torch.cuda.is_available() else 'cpu')

    data_specs = DataConfig.from_args(args)
    (training_data, validation_data, test_data), conditions = build_training_dataset(args, device=device, test=True)
    conditions = conditions.to(device=device, dtype=torch.float32)
    training_loader = BatchLoader(training_data, batch_size=args['training:batch_size'], drop_last=False, shuffle=True)
    validation_loader = BatchLoader(validation_data, batch_size=args['training:batch_size'], drop_last=False, shuffle=False)
    test_loader = BatchLoader(test_data, batch_size=args['training:batch_size'], drop_last=False, shuffle=False)

    experiment, run = output_factory.build_storage(args)

    m = 0
    while m < args['training:ensemble_size']:
        print(f'[INFO] Entering training loop {m + 1}')

        writer = run.get_tensorboard_summary()
        writer.add_text('params', json.dumps(args, indent=4, sort_keys=True), 0)

        best_loss = None
        counter = 0

        loss_function = loss_factory.build_loss(args).to(device)

        model = model_factory.build_model(
            args, *conditions.shape,
            data_specs.num_channels(),
            loss_function.in_channels()
        )
        model = model.to(device)

        print('[INFO] Model:')
        print(model)

        pretraining = Pretraining.from_args(args, prepare_data)
        if pretraining.num_epochs > 0:
            pretraining.run(args, model, training_loader, conditions, device)

        optimizer = optim_factory.build_optimizer(args, model)
        trainer = Trainer(
            args,
            model,
            loss_function,
            optimizer,
            sched_factory.build_scheduler(args, optimizer),
            prepare_data, writer, device
        )

        timer = Timer()
        timer.start()
        for e in range(args['training:num_epochs']):
            train_loss = trainer.train_epoch(e, training_loader, conditions)
            val_loss = trainer.validate_epoch(e, validation_loader, conditions)
            trainer.validate_epoch(e, test_loader, conditions, tag='test')
            trainer.advance_scheduler(e, val_loss)

            if model['feature_selector'] is not None:
                print(f'[INFO] Fraction of active features: {model["feature_selector"].get_valid_fraction()}')

            if best_loss is None or val_loss < best_loss:
                print(f'[INFO] Storing checkpoint with new best loss of {val_loss}')
                run.save_checkpoint(model, 'best_model_member{:02d}.pth'.format(m))
                best_loss = val_loss
                counter = 0
            else:
                counter = counter + 1

            writer.add_scalar('loss/best', best_loss, e + 1)
            writer.add_scalar('patience', counter, e + 1)

            if counter >= args['training:patience']:
                print('[INFO] Hit patience counter.')
                break
            if np.isnan(best_loss):
                print('[INFO] Encountered nan loss')
                break
        if np.isnan(best_loss):
            print(f'[INFO] Repeating training loop {m} due to nan loss')
        else:
            timer.stop()
            print(f'[INFO] Leaving training loop {m} with final best loss {best_loss}')
            model = run.load_checkpoint('best_model_member{:02d}.pth'.format(m), map_location=device)
            trainer.model = model
            final_loss = trainer.validate_epoch(args['training:num_epochs'] + 1, validation_loader, conditions, tag=None)
            writer.add_scalar('final_loss', final_loss)
            writer.add_scalar('train_time', timer.read())
            writer.add_scalar('num_params', sum([p.numel() for p in model.parameters() if p.requires_grad]))
            m = m + 1
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()
