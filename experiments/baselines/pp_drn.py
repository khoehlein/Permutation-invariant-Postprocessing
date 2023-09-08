import json
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from torch import nn, Tensor
from torch.nn import Embedding

from experiments.baselines.common import initiate, BaseModel, prepare_data, \
    run_single_training, build_storage
from experiments.baselines.drn_utils import crps_logistic, crps_logistic_torch, \
    crps_normal_torch, crps_normal


def _build_model(nn_ls: Dict[str, Any], n_dir_preds: int, n_stations: int, activation_out=None):
    if activation_out is None:
        activation_out = nn.Softplus()
    embedding = Embedding(n_stations, nn_ls['emb_dim'])
    lay1 = nn_ls['lay1']
    mlp = nn.Sequential(
        nn.Linear(n_dir_preds + nn_ls['emb_dim'], lay1),
        nn.Softplus(),
        nn.Linear(lay1, lay1 // 2),
        nn.Softplus(),
        nn.Linear(lay1 // 2, 2),
        activation_out,
    )
    model = BaseModel(embedding, mlp)

    def _init(m: nn.Module):
        # initialize like in tensorflow
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight.data, -0.05, 0.05)
            nn.init.uniform_(m.bias.data, -0.05, 0.05)
        if isinstance(m, Embedding):
            nn.init.uniform_(m.weight.data, -0.05, 0.05)

    model.apply(_init)
    return model


class Exponential(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)


def drn_pp(
        train: pd.DataFrame, # training data
        X: pd.DataFrame, # test data
        i_valid=None, # validation index range
        loc_id_vec: np.ndarray = None, # id oflocations (string)
        pred_vars = None, # predictors used for NN
        nn_ls = None, # training hyperparameters
        n_ens: int = 20, # ensemble size
        n_cores = None, # number of cores used in keras
        scores_ens = True, # Compute CRPS/Log score of ensemble
        scores_pp = True, # Compute CRPS/Log score of NN
        output_path = None,
        output_mode = None,
):
    X, nn_ls, pred_vars, train = initiate(X, n_cores, n_ens, nn_ls, pred_vars, scores_ens, scores_pp, train)

    hpar_ls = dict(
        n_sim=10,
        lr_adam=5.e-4,  # previously 1e-3
        n_epochs=150,
        n_patience=10,
        n_batch=64,
        emb_dim=10,
        lay1=64,
        actv="softplus",
        nn_verbose=0,
        method='logistic'
    )

    hpar_ls.update(nn_ls)
    nn_ls = hpar_ls

    (
        X_pred,
        X_train, y_train,
        X_valid, y_valid,
        n_dir_preds, n_test, n_train, n_valid
    ) = prepare_data(
        X, i_valid, loc_id_vec, pred_vars, train
    )

    log, run = build_storage(nn_ls, output_mode, output_path)

    runtime_est, runtime_pred = 0., 0.
    f = np.zeros((len(X), 2), dtype=np.float32)

    if nn_ls['method'] == 'logistic':
        loss_fn = crps_logistic_torch
        eval_fn = crps_logistic
    elif nn_ls['method'] == 'normal':
        loss_fn = crps_normal_torch
        eval_fn = crps_normal
    else:
        raise NotImplementedError()

    for i_sim in range(nn_ls['n_sim']):
        if log is not None:
            writer = run.get_tensorboard_summary()
            writer.add_text('params', json.dumps(nn_ls, indent=4, sort_keys=True), 0)
        else:
            writer = None

        # if nn_ls['method'] == 'normal':
        #     act = Exponential()
        # else:
        #     act = None
        model = _build_model(nn_ls, n_dir_preds, len(np.unique(loc_id_vec)), activation_out=None)
        print(model)
        res = None
        while res is None:
            res = run_single_training(
                model,
                loss_fn,
                X_pred,
                X_train, y_train,
                X_valid, y_valid,
                nn_ls,
                writer
            )

        runtime_est += res['runtime_est']
        runtime_pred += res['runtime_pred']
        f += res['f']

        if log is not None:
            # validation
            scores = eval_fn(res['f_valid'], y_valid).values
            print('[INFO] Storing predictions (valid)...')
            log.save_predictions(
                i_sim,
                res['f_valid'], y_valid,
                scores, 'valid'
            )
            writer.add_scalar('train_time', res['runtime_est'], i_sim)
            writer.add_scalar('final_loss', np.mean(scores), i_sim)
            writer.flush()
            writer.close()

            # test
            scores = eval_fn(res['f'], X.loc[:, 'obs'].values).values
            print('[INFO] Storing predictions (test)...')
            log.save_predictions(
                i_sim,
                res['f'], X.loc[:, 'obs'].values,
                scores, 'test'
            )
            run.save_checkpoint(model, f'best_model_{i_sim}.pth')


    f = f / nn_ls['n_sim']

    if scores_pp:
        scores_pp = eval_fn(f, X.loc[:, 'obs'].values)

    return {
        'f': f,
        'runtime_est': runtime_est,
        'runtime_pred': runtime_pred,
        'nn_ls': nn_ls,
        'pred_vars': pred_vars,
        'n_train': n_train,
        'n_valid': n_valid,
        'n_test': n_test,
        'scores_ens': None,
        'scores_pp': scores_pp,
    }
