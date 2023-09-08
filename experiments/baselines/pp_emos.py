import copy
import json
from typing import Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn, Tensor
from torch.optim import LBFGS

from experiments.helpers.timer import Timer
from experiments.baselines.common import build_storage, prepare_data, initiate
from experiments.baselines.drn_utils import crps_logistic, \
    crps_logistic_torch, crps_normal, crps_normal_torch

np.random.seed(42)


class TruncatedLogisticEMOS(nn.Module):

    def __init__(
            self,
            t_min: float = 1.e-3,
            t_max: float = 1.e3,
            t_0: float = 1.e-2,
            init_abcd=None,
            init_sigma=0.1
    ):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.t_0 = t_0
        if init_abcd is None:
            init_abcd = (np.random.rand(4) - 0.5) * init_sigma
            init_abcd[-1] += 1.
            init_abcd = init_abcd.tolist()
        self.a = nn.Parameter(torch.as_tensor(init_abcd[0]), requires_grad=True)
        self.b = nn.Parameter(torch.as_tensor(init_abcd[1]), requires_grad=True)
        self.c = nn.Parameter(torch.as_tensor(init_abcd[2]), requires_grad=True)
        self.d = nn.Parameter(torch.as_tensor(init_abcd[3]), requires_grad=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        ens_mean, ens_std = x[..., 0], x[...,1]
        loc = torch.clip(self.a  + torch.exp(self.b) * ens_mean, min=-self.t_max, max=self.t_max)
        scale = torch.clip(torch.exp(self.c + self.d * torch.log(ens_mean)), min=self.t_min, max=self.t_max)
        return torch.stack([loc, scale], dim=-1)

    def fit(
            self,
            X: Tensor, y: Tensor,
            max_steps=100, lr=1.0, reltol=0.01, patience=10,
            writer=None,
            tag=None,
            loss_fn=None,
    ):

        if tag is None:
            tag = 'loss'

        if loss_fn is None:
            loss_fn = crps_logistic_torch

        optimizer = LBFGS(self.parameters(), lr=lr)

        def closure():
            optimizer.zero_grad()
            loss = crps_logistic_torch(self.forward(X), y)
            loss.backward()
            return loss

        losses = []
        best_loss = None
        best_state = copy.deepcopy(self.state_dict())
        counter = 0
        for i in range(max_steps):
            self.train()
            optimizer.step(closure)
            with torch.no_grad():
                self.eval()
                new_loss = loss_fn(self.forward(X), y).item()
                if writer is not None:
                    writer.add_scalar(tag, new_loss, i)
                losses.append(new_loss)
                if best_loss is None or (best_loss - new_loss) / best_loss > reltol:
                    best_loss = new_loss
                    best_state = copy.deepcopy(self.state_dict())
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    break
        self.load_state_dict(best_state)
        self.eval()
        return losses

    def predict(self, data: Tensor):
        return self.forward(data)


def run_single_training(
        X_train: pd.DataFrame, y_train,  # training data
        X_valid: pd.DataFrame, y_valid,  # training data
        X_pred: pd.DataFrame,  # test data
        loss_fn,
        args: Dict[str, Any],
        writer=None, tag=None
):
    if tag is None:
        tag = 'loss'
    if args['local']:
        res = _run_local_training(
            X_pred,
            X_train, y_train,
            X_valid, y_valid,
            loss_fn,
            args, writer=writer, tag=tag
        )
    else:
        res = _run_global_training(
            X_pred,
            X_train, y_train,
            X_valid, y_valid,
            loss_fn,
            args, writer=writer, tag=tag
        )

    return res


def predict_local(model: nn.ModuleDict, X: Dict[str, np.ndarray]):
    with torch.no_grad():
        model.eval()
        predictors = torch.from_numpy(X['dir_input'])
        predictions = np.empty((len(predictors), 2))
        group_data = pd.DataFrame({
            'location': X['id_input'].ravel(),
            'month': X['month_input'].ravel()
        })
        groups = group_data.groupby(by=['location', 'month']).groups
        for key in groups.keys():
            loc_idx = groups.get(key)
            predictions[loc_idx] = model[_key_to_model_tag(key)](predictors[loc_idx]).data.cpu().numpy()
    return predictions


def _key_to_model_tag(key):
    return str(hash(key))

def _run_local_training(
        X_pred,
        X_train, y_train,
        X_valid, y_valid,
        loss_fn,
        args,
        writer=None,
        tag=None,
):
    group_data = pd.DataFrame({
        'location': X_train['id_input'].ravel(),
        'month': X_train['month_input'].ravel(),
    })
    groups = group_data.groupby(by=['location', 'month']).groups

    groups_next = group_data.copy()
    groups_next['month'] = (group_data['month'].values - 1) % 12
    groups_next = groups_next.groupby(by=['location', 'month']).groups

    groups_previous = group_data.copy()
    groups_previous['month'] = (group_data['month'].values + 1) % 12
    groups_previous = groups_previous.groupby(by=['location', 'month']).groups

    model = nn.ModuleDict({_key_to_model_tag(key): TruncatedLogisticEMOS(init_sigma=args['init_sigma']) for key in groups.keys()})
    predictors = X_train['dir_input']

    if tag is None:
        tag = 'loss'

    timer = Timer()
    timer.start()
    with tqdm.tqdm(total=len(groups.keys())) as pbar:
        for key in groups.keys():
            # build index for local/seasonal training dataset
            loc_idx = []
            for groups_ in [groups, groups_next, groups_previous]:
                loc_idx.append(groups_.get(key))
            loc_idx = np.unique(np.concatenate(loc_idx, axis=0))

            group_train = torch.from_numpy(predictors[loc_idx])
            group_y = torch.from_numpy(y_train[loc_idx])

            model[_key_to_model_tag(key)].fit(
                group_train, group_y,
                lr=args['lr_lbfgs'],
                patience=args['n_patience'],
                max_steps=args['n_epochs'],
                writer=writer,
                tag=f'{tag}/model_{key}',
                loss_fn=loss_fn,
            )
            pbar.update(1)
    timer.stop()
    runtime_est = timer.read()

    with torch.no_grad():
        # validation
        f_valid = predict_local(model, X_valid)

        # test
        timer = Timer()
        timer.start()
        f = predict_local(model, X_pred)
        timer.stop()
        runtime_pred = timer.read()

    return {
        'model': model,
        'f': f, 'f_valid': f_valid,
        'runtime_est': runtime_est,
        'runtime_pred': runtime_pred,
    }


def predict(model: Union[nn.Module, TruncatedLogisticEMOS], X: Dict[str, np.ndarray]):
    if isinstance(model, nn.ModuleDict):
        return predict_local(model, X)
    else:
        return model(torch.from_numpy(X['dir_input'])).data.cpu().numpy()


def _run_global_training(
        X_pred,
        X_train, y_train,
        X_valid, y_valid,
        loss_fn,
        args, writer=None, tag=None
):
    if tag is None:
        tag = 'loss'
    model = TruncatedLogisticEMOS(init_sigma=args['init_sigma'])
    timer = Timer()
    timer.start()
    print('[INFO] Fitting model.')
    model.fit(
        torch.from_numpy(X_train['dir_input']), torch.from_numpy(y_train),
        lr=args['lr_lbfgs'], patience=args['n_patience'], max_steps=args['n_epochs'],
        writer=writer, tag=tag, loss_fn=loss_fn,
    )
    timer.stop()
    runtime_est = timer.read()

    with torch.no_grad():
        # validation
        f_valid = model(torch.from_numpy(X_valid['dir_input'])).data.cpu().numpy()

        # test
        timer = Timer()
        timer.start()
        f = model(torch.from_numpy(X_pred['dir_input'])).data.cpu().numpy()
        timer.stop()

    runtime_pred = timer.read()
    return {
        'model': model,
        'f': f, 'f_valid': f_valid,
        'runtime_est': runtime_est,
        'runtime_pred': runtime_pred,
    }


def emos_pp(
        train: pd.DataFrame,
        X: pd.DataFrame,
        i_valid=None,
        loc_id_vec: np.ndarray = None,
        pred_vars = None,
        args = None,
        n_ens=20,
        n_cores=None,
        scores_pp=True,
        scores_ens=True,
        output_path = None,
        output_mode = None,
):

    (
        X, nn_ls, pred_vars, train
    ) = initiate(X, n_cores, n_ens, args, pred_vars, scores_ens, scores_pp, train)

    (
        X_pred,
        X_train, y_train,
        X_valid, y_valid,
        n_dir_preds, n_test, n_train, n_valid,
    ) = prepare_data(X, i_valid, loc_id_vec, pred_vars, train, rescale=False)

    log, run = build_storage(args, output_mode, output_path)

    runtime_est, runtime_pred = 0., 0.
    f = np.zeros((n_test, 2), dtype=np.float32)

    if log is not None:
        writer = run.get_tensorboard_summary()
        writer.add_text('params', json.dumps(args, indent=4, sort_keys=True), 0)
    else:
        writer = None

    if args['method'] == 'logistic':
        loss_fn = crps_logistic_torch
        eval_fn = crps_logistic
    elif args['method'] == 'normal':
        loss_fn = crps_normal_torch
        eval_fn = crps_normal
    else:
        raise NotImplementedError()


    for i_sim in range(args['n_sim']):

        res = run_single_training(
            X_train, y_train,
            X_valid, y_valid,
            X_pred,
            loss_fn,
            args, writer=writer,
            tag=f'loss/sim_{i_sim}',
        )

        runtime_est += res['runtime_est']
        runtime_pred += res['runtime_pred']
        f += res['f']


        if log is not None:
            # validation
            scores = eval_fn(res['f_valid'], y_valid).values
            print('[INFO] Storing predictions (validation)...')
            log.save_predictions(
                i_sim,
                res['f_valid'], y_valid,
                scores, 'valid'
            )
            writer.add_scalar('train_time', res['runtime_est'], i_sim)
            writer.add_scalar('final_loss', np.mean(scores), i_sim)

            # test
            scores = eval_fn(res['f'], X.loc[:, 'obs'].values).values
            writer.add_scalar('test_loss', np.mean(scores), i_sim)
            print('[INFO] Storing predictions (test)...')
            log.save_predictions(
                i_sim,
                res['f'], X.loc[:, 'obs'].values,
                scores, 'test'
            )
            run.save_checkpoint(res['model'], f'best_model_{i_sim}.pth')
            writer.flush()
    if log is not None:
        writer.close()

    f = f / args['n_sim']

    if scores_pp:
        scores_pp = crps_logistic(f, X.loc[:, 'obs'].values)

    return {
        'f': f,
        'runtime_est': runtime_est,
        'runtime_pred': runtime_pred,
        'n_train': len(train),
        'n_valid': 0,
        'n_test': len(X),
        'scores_ens': None,
        'scores_pp': scores_pp
    }


def _test():
    with torch.no_grad():
        X = torch.exp(torch.randn(10000, 2))
        y = torch.exp(torch.randn(10000, 1))

    from matplotlib import pyplot as plt
    plt.figure()
    ax = plt.gca()

    for _ in range(6):
        abcd = torch.randn(4) * 0.1
        abcd[-1] += 1.
        model= TruncatedLogisticEMOS(init_abcd=[x for x in abcd])
        losses = model.fit(X, y)
        ax.plot(losses)
    ax.set(yscale='log')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _test()
