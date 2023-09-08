import argparse
import torch
from experiments.euppbench_reforecasts.predict import Predict as BasePredict
from data.cosmo_de import build_training_dataset as build_gusts_data


torch.set_num_threads(4)

class PredictGusts(BasePredict):

    def _load_data(self, args, device):
        data, conditions = build_gusts_data(args, test=True, device=device)
        data = {key: val for key, val in zip(['train', 'valid', 'test'], data)}[self.dataset]
        return data, conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('-r', action='store_true', dest='recurse')
    parser.set_defaults(recurse=False)
    args = vars(parser.parse_args())

    with torch.no_grad():
        for dataset in ['test', 'valid']:
            print(f'[INFO] Predicting for dataset "{dataset}"')
            predict = PredictGusts(dataset)
            if args['recurse']:
                predict.predict_or_recurse(args['path'])
            else:
                predict.from_run_path(args['path'])


if __name__ == '__main__':
    main()
