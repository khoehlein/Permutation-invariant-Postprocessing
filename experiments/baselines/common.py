import copy
import datetime
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import Embedding
from torch.optim import Adam
from torch.utils.data import TensorDataset
import tqdm

from data.utils import BatchLoader
from utils.automation.storage import MultiRunExperiment
from utils.progress import WelfordStatisticsTracker


def initiate(X, n_cores, n_ens, nn_ls, pred_vars, scores_ens, scores_pp, train):
    if pred_vars is None:
        pred_vars = ['ens_mean', 'ens_sd', 'location']
    if nn_ls is None:
        nn_ls = {}
    train_vars = ['obs', 'location', *pred_vars]
    test_vars = ['location', *pred_vars]
    ens_vars = [f'ens_{i}' for i in range(1, n_ens + 1)]
    if scores_ens:
        missing_ens_vars = _find_missing_variables(X.columns, ens_vars)
        if len(missing_ens_vars) > 0:
            scores_ens = False
    if scores_pp or scores_ens:
        test_vars = [*test_vars, 'obs']
    missing_train_vars = _find_missing_variables(train.columns, train_vars)
    if len(missing_train_vars) > 0:
        print(f'[INFO] Training data does not include relevant variables: {missing_train_vars}')
    missing_test_vars = _find_missing_variables(X.columns, test_vars)
    if len(missing_train_vars) > 0:
        print(f'[INFO] Test data does not include relevant variables: {missing_test_vars}')
    train = train.loc[:, train_vars]
    if scores_ens:
        X = X.loc[:, _unique(test_vars + ens_vars)]
    else:
        X = X.loc[:, test_vars]
    if train.isnull().values.any():
        print('[INFO] Encountered NaNs in training data!')
        train = train.dropna(axis=0)
    if X.isnull().values.any():
        print('[INFO] Encountered NaNs in test data!')
        X = X.dropna(axis=0)
    if 'ens_sd' in pred_vars:
        if (train['ens_sd'] < 0).values.any():
            print('[INFO] At least one ensemble sd in training data is negative!')
        if (X['ens_sd'] < 0).values.any():
            print('[INFO] At least one ensemble sd in test data is negative!')
    if n_cores is not None:
        raise ValueError(
            '[ERROR] Use environment variables "OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS" and "TF_NUM_INTEROP_THREADS" to limit core usage.')
    t_c = 0.
    return X, nn_ls, pred_vars, train


def build_storage(nn_ls, output_mode, output_path):
    if output_path is not None:
        if output_mode == 'exp':
            experiment = MultiRunExperiment(output_path)
            run = experiment.create_new_run()
        elif output_mode == 'run':
            experiment, run = MultiRunExperiment.from_run_path(output_path, return_run=True)
        else:
            raise Exception()
        run.add_parameter_settings(nn_ls)
        log = LogDirectory(os.path.join(run.get_evaluation_path(), 'log'))
    else:
        log = None
        run = None
    return log, run


class BaseModel(nn.Module):

    def __init__(self, embedding: Embedding, mlp: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.mlp = mlp

    def forward(self, dir_input: Tensor, id_input: Tensor):
        if len(id_input.shape) > 1:
            assert len(id_input.shape) == 2
            id_input = id_input.squeeze(-1)
        emb = self.embedding(id_input)
        features = torch.cat([dir_input, emb], dim=-1)
        return self.mlp(features)


def prepare_data(X, i_valid, loc_id_vec, pred_vars, train, rescale=True):
    if i_valid is None:
        r_valid = 0.25 # this differs from original!
        i_train = np.arange(math.floor(len(train) * (1. - r_valid)))
        i_valid = np.arange(i_train[-1] + 1, len(train))
    else:
        i_train = np.asarray(list(set(range(len(train))).difference(set(i_valid))))
    n_train = len(i_train)
    n_valid = len(i_valid)
    n_test = len(X)
    # Constant predictors are not removed here, but are not to be expected anyways...
    dir_preds = [p for p in pred_vars if p not in ['location', 'month']]
    n_dir_preds = len(dir_preds)

    X_train = train.loc[i_train, dir_preds].values
    scales = {}
    if rescale:
        X_train, scales = _scale(X_train)
    X_train = {
        'dir_input': X_train,
        'id_input': train.loc[i_train, 'location'].values[:, None],
    }
    if 'month' in pred_vars:
        X_train.update({'month_input': train.loc[i_train, 'month'].values[:, None]})
    y_train = train.loc[i_train, 'obs'].values

    X_valid = train.loc[i_valid, dir_preds].values
    if rescale:
        X_valid, _ = _scale(X_valid, **scales)
    X_valid = {
        'dir_input': X_valid,
        'id_input': train.loc[i_valid, 'location'].values[:, None],
    }
    if 'month' in pred_vars:
        X_valid.update({'month_input': train.loc[i_valid, 'month'].values[:, None]})
    y_valid = train.loc[i_valid, 'obs'].values
    # x_id = np.argmax(
    #     X.loc[:, 'location'].values.ravel()[:, None] == np.asarray(loc_id_vec).ravel()[None, :], axis=-1)[:, None]
    x_id = X.loc[:, 'location'].values[:, None]
    X_pred = X.loc[:, dir_preds].values
    if rescale:
        X_pred, _ = _scale(X_pred, **scales)
    X_pred = {
        'dir_input': X_pred,
        'id_input': x_id,
    }
    if 'month' in pred_vars:
        X_pred.update({'month_input': X.loc[:, 'month'].values[:, None]})
    return (
        X_pred,
        X_train, y_train,
        X_valid, y_valid,
        n_dir_preds, n_test, n_train, n_valid
    )


class EarlyStopping(object):

    def __init__(self, patience: int = None):
        assert patience is not None
        self.patience = patience
        self.best_metric = None
        self.counter = 0
        self.state = None

    def update(self, metric: float, model: nn.Module):
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            self.state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return self
        self.counter += 1
        return self

    def exit_condition_met(self):
        return self.counter > self.patience


class LogDirectory(object):

    def __init__(self, path: str):
        self.path = os.path.join(os.path.abspath(path), datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S.%f'))
        os.makedirs(self.path, exist_ok=True)

    def save_predictions(self, m: int,  f: np.ndarray, obs: np.ndarray, crps: np.ndarray, tag: str):
        np.savez(
            os.path.join(self.path, f'member_{m}_{tag}.npz'),
            predictions=f, obs=obs, crps=crps
        )


def _check_if_ensemble_is_available(X: pd.DataFrame, n_ens: int):
    ens_vars = [f'ens_{i}' for i in range(1, n_ens + 1)]
    return (len(_find_missing_variables(X.columns, ens_vars)) == 0)


def _find_missing_variables(source, variables):
    return list(set(variables).difference(set(source)))


def _unique(x):
    return list(set(x))


def _scale(data: np.ndarray, tr_center=None, tr_scale=None):
    if tr_center is None:
        tr_center = np.mean(data, axis=0, keepdims=True)
    if tr_scale is None:
        tr_scale = np.std(data, axis=0, keepdims=True, ddof=1)
    return ((data - tr_center) / tr_scale, {'tr_center': tr_center, 'tr_scale': tr_scale})


def run_single_training(
        model,
        loss_fn,
        X_pred,
        X_train, y_train,
        X_valid, y_valid,
        nn_ls,
        writer
):
    try:
        lr = nn_ls['lr_adam']
        if lr == -1:
            lr = 0.001
        optimizer = Adam(model.parameters(), lr=lr)

        start_tm = time.time()

        fit(
            model, optimizer, loss_fn,
            X_train, y_train, nn_ls['n_epochs'],
            batch_size=nn_ls['n_batch'],
            validation_data=[X_valid, y_valid],
            verbose=nn_ls['nn_verbose'],
            early_stopping=EarlyStopping(patience=nn_ls['n_patience']),
            writer=writer,
        )

        end_tm = time.time()

        runtime_est = end_tm - start_tm

        start_tm = time.time()

        with torch.no_grad():
            f = predict(model, X_pred, batch_size=nn_ls['n_batch']).data.cpu().numpy()
            f_valid = predict(model, X_valid, batch_size=nn_ls['n_batch']).data.cpu().numpy()

        end_tm = time.time()

        runtime_pred = end_tm - start_tm

        output = {
            'f': f,
            'f_valid': f_valid,
            'runtime_est': runtime_est,
            'runtime_pred': runtime_pred,
        }
    except Exception as e:
        print(e)
        output = None
    return output



def _to_tensor_dataset(dir_input=None, id_input=None, obs=None,  device=None):
    tensors = [
        torch.from_numpy(dir_input).to(dtype=torch.float32, device=device),
        torch.from_numpy(id_input).to(dtype=torch.long, device=device),
    ]
    if obs is not None:
        tensors.append(torch.from_numpy(obs).to(dtype=torch.float32, device=device))
    return TensorDataset(*tensors)


def predict(model, X, batch_size=1, verbose=True):
    model.eval()
    dataset = _to_tensor_dataset(**X)
    loader = BatchLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    def _predict(pbar):
        i = 0
        predictions = None
        for batch in loader:
            dir_input, id_input = batch
            pred_batch = model(dir_input, id_input)
            num_preds = len(pred_batch)
            if predictions is None:
                predictions = torch.zeros(len(dataset), pred_batch.shape[-1])
            predictions[i:(i + num_preds)] = pred_batch
            i = i + num_preds
            if pbar is not None:
                pbar.update(1)
        return predictions
    
    if verbose:
        with tqdm.tqdm(total=len(loader)) as pbar:
            predictions = _predict(pbar)
    else:
        predictions = _predict(None)
    return predictions


def fit(
        model, optimizer, loss_fn,
        x, y,
        epochs,
        batch_size=1,
        validation_data=None,
        verbose=True,
        early_stopping: EarlyStopping = None,
        writer=None
):
    data_train = _to_tensor_dataset(**x, obs=y)
    train_loader = BatchLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False)

    def train():
        model.train()
        tracker = WelfordStatisticsTracker()
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for i, batch in enumerate(train_loader):
                model.zero_grad()
                dir_input, id_input, obs = batch
                pred_batch = model(dir_input, id_input)
                loss = loss_fn(pred_batch, obs)
                loss.backward()
                optimizer.step()
                tracker.update(loss.item(), weight=len(pred_batch))
                pbar.update(1)
        return tracker.mean()

    def validate():
        with torch.no_grad():
            predictions = predict(model, validation_data[0], batch_size=batch_size)
            obs = torch.as_tensor(validation_data[1])
            crps = loss_fn(predictions, obs)
        return crps.item()

    for epoch in range(epochs):
        train_loss = train()
        print(f'[INFO] Train loss: {train_loss}')
        val_loss = validate()
        print(f'[INFO] Valid loss: {val_loss}')
        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch + 1)
            writer.add_scalar('loss/valid', val_loss, epoch + 1)

        if early_stopping is not None:
            early_stopping.update(val_loss, model)
            if writer is not None:
                writer.add_scalar('loss/best', early_stopping.best_metric, epoch + 1)
                writer.add_scalar('patience', early_stopping.counter, epoch + 1)
            if early_stopping.exit_condition_met():
                break


    if early_stopping is not None:
        model.load_state_dict(early_stopping.state)

    return model


class Coverage(object):

    def __init__(self, alpha: float):
        self.alpha = alpha

    def from_pit(self, pit: np.ndarray):
        return np.mean(np.logical_and(self.alpha / 2. <= pit, pit <= (1. - self.alpha / 2.)))

    def from_ranks(self, ranks: np.ndarray, max_rank: int = None):
        if max_rank is None:
            max_rank = np.max(ranks)
        min_cover = self.alpha / 2. * max_rank
        max_cover = max_rank - min_cover
        return np.mean(np.logical_and(ranks >= min_cover, ranks <= max_cover))
