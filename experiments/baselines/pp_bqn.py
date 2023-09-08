import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import scipy.special
import torch
from torch import nn
from torch.nn import Embedding
# from inference.evaluation.metrics import fn_scores_ens
from experiments.baselines.common import (
    initiate, prepare_data, build_storage, BaseModel, run_single_training
)


def dbinom(x: np.ndarray, size, prob):
    return scipy.stats.binom.pmf(x, size, prob)


def crps_sample(ens, obs):
    # CRPS_PWM from https://link.springer.com/article/10.1007/s11004-017-9709-7
    n_ens = ens.shape[-1]
    ens = np.sort(ens, axis=-1)
    a = np.mean(np.abs(ens - obs[:, None]), axis=-1)
    b0 = np.mean(ens, axis=-1)
    b1 = np.mean(np.arange(n_ens) * ens, axis=-1) / (n_ens - 1)
    crps = a + b0 - 2. * b1
    return crps


def _build_model(nn_ls: Dict[str, Any], n_dir_preds: int, n_stations: int):
    embedding = Embedding(n_stations, nn_ls['emb_dim'])
    lay1 = nn_ls['lay1']
    mlp = nn.Sequential(
        nn.Linear(n_dir_preds + nn_ls['emb_dim'], lay1),
        nn.Softplus(),
        nn.Linear(lay1, lay1 // 2),
        nn.Softplus(),
        nn.Linear(lay1 // 2, nn_ls['p_degree'] + 1),
        nn.Softplus(),
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


def bern_quants(alpha: np.ndarray, q_levels: np.ndarray):
    p_degree = alpha.shape[-1] - 1
    B = dbinom(np.arange(p_degree + 1)[:, None], p_degree, q_levels[None, :])
    res = np.matmul(alpha, B)
    return res


# def _compute_model_scores(coeff_bern, q_levels, observations, n_ens):
#     q = bern_quants(coeff_bern, q_levels)
#     scores_pp = fn_scores_ens(q, observations, n_ens_ref=n_ens)
#     # Transform ranks to n_(ens + 1) bins (for multiples of (n_ens + 1) exact)
#     if q.shape[-1] != n_ens:
#         scores_pp['rank'] = np.ceil(scores_pp['rank'] * (n_ens + 1) / (q.shape[-1] + 1))
#     scores_pp['e_me'] = np.mean(coeff_bern, axis=-1) - observations
#     return scores_pp


def _compute_model_scores(coeff_bern, q_levels, observations):
    q = bern_quants(coeff_bern, q_levels)
    return pd.Series(crps_sample(q, observations), name='CRPS')


def bqn_pp(
        train: pd.DataFrame,  # training data
        X: pd.DataFrame,  # test data
        i_valid=None,  # validation index range
        loc_id_vec: np.ndarray = None,  # id of locations (string)
        pred_vars=None,  # predictors used for NN
        q_levels=None, # Quantile levels used for output and evaluation (n_q probability vector)
        nn_ls=None,  # training hyperparameters
        n_ens: int = 20,  # ensemble size
        n_cores=None,  # number of cores used in keras
        scores_ens=True,  # Compute CRPS/Log score of ensemble
        scores_pp=True,  # Compute CRPS/Log score of NN
        output_path=None,
        output_mode=None,
):

    X, nn_ls, pred_vars, train = initiate(X, n_cores, n_ens, nn_ls, pred_vars, scores_ens, scores_pp, train)

    # If not given use equidistant quantiles (multiple of ensemble coverage, incl. median)
    if q_levels is None:
        q_levels = np.arange(1., 6. * (n_ens + 1)) / (6 * (n_ens + 1))

    hpar_ls = dict(
        p_degree = 12,
        n_q = 99,
        n_sim=10,
        lr_adam=5.e-4,
        n_epochs=150,
        n_patience=10,
        n_batch=64,
        emb_dim=10,
        lay1=48,
        actv="softplus",
        nn_verbose=0
    )

    hpar_ls.update(nn_ls)
    nn_ls = hpar_ls

    q_levels_loss = np.linspace(1, nn_ls['n_q'] + 1) / (nn_ls['n_q'] + 1)

    B = dbinom(np.arange(nn_ls['p_degree'] + 1)[:, None], nn_ls['p_degree'], q_levels_loss[None, :])
    B = torch.as_tensor(B).to(torch.float32)
    q_levels_loss = torch.as_tensor(q_levels_loss).to(torch.float32)

    def qt_loss(y_pred, y_true):
        q = torch.matmul(torch.cumsum(y_pred, dim=-1), B)
        err = y_true.unsqueeze(-1) - q
        e1 = err * q_levels_loss
        e2 = err * (q_levels_loss - 1.)
        return torch.mean(2. * torch.maximum(e1, e2))

    (
        X_pred,
        X_train, y_train,
        X_valid, y_valid,
        n_dir_preds, n_test, n_train, n_valid,
    ) = prepare_data(X, i_valid, loc_id_vec, pred_vars, train)

    log, run = build_storage(nn_ls, output_mode, output_path)

    runtime_est, runtime_pred = 0., 0.
    coeff_bern = np.zeros((len(X), nn_ls['p_degree'] + 1), dtype=np.float32)

    for i_sim in range(nn_ls['n_sim']):
        if log is not None:
            writer = run.get_tensorboard_summary()
            writer.add_text('params', json.dumps(nn_ls, indent=4, sort_keys=True), 0)
        else:
            writer = None

        model = _build_model(nn_ls, n_dir_preds, len(np.unique(loc_id_vec)))
        print(model)
        res = run_single_training(
            model,
            qt_loss,
            X_pred,
            X_train, y_train,
            X_valid, y_valid,
            nn_ls,
            writer
        )

        runtime_est += res['runtime_est']
        runtime_pred += res['runtime_pred']
        coeff_bern += res['f']

        if log is not None:
            # validation
            scores = _compute_model_scores(np.cumsum(res['f_valid'], axis=-1), q_levels, y_valid)
            print('[INFO] Storing predictions (valid)...')
            log.save_predictions(
                i_sim,
                np.cumsum(res['f_valid'], axis=-1), y_valid,
                scores, 'valid'
            )
            writer.add_scalar('train_time', res['runtime_est'], i_sim)
            writer.add_scalar('final_loss', np.mean(scores), i_sim)
            writer.flush()
            writer.close()

            # test
            scores = _compute_model_scores(np.cumsum(res['f'], axis=-1), q_levels, X['obs'].values)
            print('[INFO] Storing predictions (test)...')
            log.save_predictions(
                i_sim,
                np.cumsum(res['f'], axis=-1), X.loc[:, 'obs'].values,
                scores, 'test'
            )
            run.save_checkpoint(model, f'best_model_{i_sim}.pth')

    coeff_bern = coeff_bern / nn_ls['n_sim']
    coeff_bern = np.cumsum(coeff_bern, axis=-1)

    if scores_pp:
        scores_pp = _compute_model_scores(coeff_bern, q_levels, X['obs'].values)

    return {
        'f': bern_quants(coeff_bern, q_levels),
        'alpha': coeff_bern,
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
