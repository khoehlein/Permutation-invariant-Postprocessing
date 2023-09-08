import argparse

import torch

from data.utils import BatchLoader
from experiments.euppbench_reforecasts.predict import Predict
from data.euppbench.forecasts import build_training_dataset as build_eupp_data
from model.loss import factory

torch.set_num_threads(4)

class ForecastPredict(Predict):

    def __init__(self):
        super().__init__('forecasts')

    def load_data(self, args, device, run):
        data, conditions = build_eupp_data(args, test=True, device=device)
        loss = factory.build_loss(args)
        checkpoints = run.list_checkpoints(sort_output=True)
        data = data[-1]
        loader = BatchLoader(data, batch_size=args['training:batch_size'], shuffle=False, drop_last=False)
        return checkpoints, conditions, data, loader, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('-r', action='store_true', dest='recurse')
    parser.set_defaults(recurse=False)
    args = vars(parser.parse_args())

    with torch.no_grad():
        print(f'[INFO] Predicting for dataset "forecasts"')
        predict = ForecastPredict()
        if args['recurse']:
            predict.predict_or_recurse(args['path'])
        else:
            predict.from_run_path(args['path'])


if __name__ == '__main__':
    main()
