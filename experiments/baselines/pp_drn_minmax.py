import json
import math
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import tqdm

from torch import nn, Tensor
from torch.distributions import Binomial
from torch.nn import Embedding
from torch.optim import Adam

from data.utils import BatchLoader
from experiments.baselines.common import initiate, BaseModel, prepare_data, \
    build_storage, EarlyStopping, predict, _to_tensor_dataset
from experiments.baselines.drn_utils import crps_logistic, crps_logistic_torch
from model.loss.losses import BernsteinWeights, TruncatedLogisticDistribution
from utils.progress import WelfordStatisticsTracker


def _build_generator(nn_ls: Dict[str, Any], n_dir_preds: int, n_stations: int):
    embedding = Embedding(n_stations, nn_ls['emb_dim'])
    lay1 = nn_ls['lay1']
    mlp = nn.Sequential(
        nn.Linear(n_dir_preds + nn_ls['emb_dim'], lay1),
        nn.Softplus(),
        nn.Linear(lay1, lay1 // 2),
        nn.Softplus(),
        nn.Linear(lay1 // 2, 2),
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



def _build_discriminator(nn_ls: Dict[str, Any], n_dir_preds: int, n_stations: int):
    embedding = Embedding(n_stations, nn_ls['emb_dim'])
    lay1 = nn_ls['lay1']
    mlp = nn.Sequential(
        nn.Linear(n_dir_preds + nn_ls['emb_dim'], lay1),
        nn.Softplus(),
        nn.Linear(lay1, lay1 // 2),
        nn.Softplus(),
        nn.Linear(lay1 // 2, nn_ls['p_degree_discriminator']),
        BernsteinWeights(nn_ls['p_degree_discriminator'], normalize=True, pad_zero=True),
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


class BernsteinDistribution(object):

    def __init__(self, weights: Tensor):
        self.weights = weights
        self.degree = weights.shape[-1] - 1

    def _bernoulli_polynomials(self, p: Tensor):
        k = torch.arange(self.degree + 1)[None, :]
        Bx = torch.exp(Binomial(self.degree, torch.reshape(p, (-1, 1))).log_prob(k))
        return Bx

    def cdf(self, x: Tensor) -> Tensor:
        Bx = self._bernoulli_polynomials(x)
        return torch.sum(self.weights * Bx, dim=-1)

    def pdf(self, x: Tensor) -> Tensor:
        cdf = self.cdf(x)
        pdf = torch.autograd.grad(cdf, x, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True)
        return pdf[0]

    def log_pdf(self, x: Tensor) -> Tensor:
        return torch.log(self.pdf(x))


class InvertibleNorm(nn.Module):

    def __init__(self, channels: int, momentum=0.999):
        super().__init__()
        self.momentum = momentum
        # self.register_buffer('log_scale', torch.zeros(channels))
        self.log_scale = math.log(273.15)
        self._init_complete = False

    def forward(self, x: Tensor):
        # if self.training:
        #     with torch.no_grad():
        #         log_scale = torch.log(torch.mean(torch.abs(x), dim=0))
        #         if self._init_complete:
        #             self.log_scale.data = self.momentum * self.log_scale.data + (1. - self.momentum) * log_scale
        #         else:
        #             self.log_scale.data = log_scale
        #             self._init_complete = True
        return x * math.exp(- self.log_scale)

    def inverse(self, x: Tensor):
        return x * math.exp(self.log_scale)


class MinMaxModel(nn.Module):

    def __init__(self, generator: BaseModel, discriminator: BaseModel):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.norm = InvertibleNorm(1)

    def _toggle_gradients(self, module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad
        return self

    def toggle_discriminator_gradients(self, requires_grad: bool):
        return self._toggle_gradients(self.discriminator, requires_grad)

    def toggle_generator_gradients(self, requires_grad: bool):
        return self._toggle_gradients(self.generator, requires_grad)

    def discriminator_loss(self, dir_input: Tensor, id_input: Tensor, obs: Tensor) -> Tensor:
        with torch.no_grad():
            prediction = self.generator(dir_input, id_input)
            obs_red = self.norm(torch.reshape(obs, (-1, 1)))
            p = TruncatedLogisticDistribution(prediction[:, 0], prediction[:, 1]).cdf(obs_red.flatten())
            p = p.detach()
        p.requires_grad = True
        weights = self.discriminator(dir_input, id_input)[0]
        deviance = torch.mean(torch.var(torch.diff(weights, dim=-1), dim=-1))
        log_p = BernsteinDistribution(weights).log_pdf(p)
        return - torch.mean(log_p) + 100. * deviance # minimize negative log-likelihood

    def generator_loss(self, dir_input: Tensor, id_input: Tensor, obs: Tensor) -> Tensor:
        prediction = self.generator(dir_input, id_input)
        obs_red = self.norm(torch.reshape(obs, (-1,)))
        p = TruncatedLogisticDistribution(prediction[:, 0], prediction[:, 1]).cdf(obs_red.flatten())
        weights = self.discriminator(dir_input, id_input)
        log_p = BernsteinDistribution(weights[0]).log_pdf(p)
        return torch.mean(log_p) # minimize log-likelihood

    def forward(self, dir_inputs: Tensor, id_inputs: Tensor) -> Tensor:
        return self.norm.inverse(self.generator(dir_inputs, id_inputs))


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

    loss_fn = crps_logistic_torch
    eval_fn = crps_logistic

    for i_sim in range(nn_ls['n_sim']):
        if log is not None:
            writer = run.get_tensorboard_summary()
            writer.add_text('params', json.dumps(nn_ls, indent=4, sort_keys=True), 0)
        else:
            writer = None

        model = _build_model(nn_ls, n_dir_preds, loc_id_vec)
        res = run_single_training(
            model, loss_fn,
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
        scores_pp = crps_logistic(f, X.loc[:, 'obs'].values)

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


def _build_model(nn_ls, n_dir_preds, loc_id_vec):
    n_loc = len(np.unique(loc_id_vec))
    model = MinMaxModel(
        _build_generator(nn_ls, n_dir_preds, n_loc),
        _build_discriminator(nn_ls, n_dir_preds, n_loc),
    )
    print(model)
    return model


def run_single_training(
        model, loss_fn,
        X_pred,
        X_train, y_train,
        X_valid, y_valid,
        nn_ls,
        writer
):
    lr = nn_ls['lr_adam']
    if lr == -1:
        lr = 0.001
    optimizer = Adam(model.parameters(), lr=lr)

    start_tm = time.time()

    fit(
        model, optimizer, loss_fn,
        X_train, y_train, nn_ls['n_epochs'],
        batch_size=nn_ls['n_batch'],
        discriminator_updates=nn_ls['n_discriminator_updates'],
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

    return {
        'f': f,
        'f_valid': f_valid,
        'runtime_est': runtime_est,
        'runtime_pred': runtime_pred,
    }


def fit(
        model, optimizer, loss_fn,
        x, y,
        epochs,
        batch_size=1,
        discriminator_updates=1,
        validation_data=None,
        verbose=True,
        early_stopping: EarlyStopping = None,
        writer=None
):
    data_train = _to_tensor_dataset(**x, obs=y)
    train_loader = BatchLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False)

    def train():
        model.train()
        tracker_gen = WelfordStatisticsTracker()
        tracker_dis = WelfordStatisticsTracker()
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for i, batch in enumerate(train_loader):
                model.zero_grad()
                if i % (discriminator_updates + 1) != 0:
                    model.toggle_discriminator_gradients(True)
                    model.toggle_generator_gradients(False)
                    loss = model.discriminator_loss(*batch)
                    tracker = tracker_dis
                else:
                    model.toggle_discriminator_gradients(False)
                    model.toggle_generator_gradients(True)
                    loss = model.generator_loss(*batch)
                    tracker = tracker_gen
                tracker.update(loss.item(), weight=len(batch[0]))
                loss.backward()
                optimizer.step()
                pbar.update(1)
        return tracker_gen.mean(), tracker_dis.mean()

    def validate():
        model.eval()
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
            writer.add_scalar('loss/train/gen', train_loss[0], epoch + 1)
            writer.add_scalar('loss/train/dis', train_loss[1], epoch + 1)
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
