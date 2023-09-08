import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.progress import WelfordStatisticsTracker


class Trainer(object):

    def __init__(self, args, model, loss_function, optimizer, scheduler, preprocess, writer, device):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_function
        self.preprocess = preprocess
        self.writer = writer
        self.device = device

    def evaluate_model(self, predictors):
        predictions = self.model['model'](*predictors)
        return predictions

    def train_epoch(self, e: int, loader, conditions):
        print('[INFO] Training model')
        tracker = WelfordStatisticsTracker()
        model = self.model
        model.train()
        with tqdm.tqdm(total=len(loader)) as pbar:
            for i, batch in enumerate(iter(loader)):
                model.zero_grad()
                predictors, observations = self.preprocess(batch, conditions, model, self.device, use_dequant=True)
                predictions = self.evaluate_model(predictors)
                loss = self.loss_func(predictions, observations)
                if self.args['model:feature_selector:weight'] > 0.:
                    p = model['feature_selector'].compute_penalty(
                        weight=self.args['model:feature_selector:weight'] * len(observations) / len(loader.dataset))
                    loss = loss + p
                if self.args['model:merger:feature_selector:weight'] > 0:
                    p = model['model'].merger.compute_penalty(
                        weight=self.args['model:merger:feature_selector:weight'] * len(observations) / len(loader.dataset))
                    loss = loss + p
                loss.backward()
                self.optimizer.step()
                pbar.update()
                tracker.update(loss.item() / len(observations), weight=len(observations))
            pbar.close()
        train_loss = tracker.mean()
        self.writer.add_scalar('loss/train', train_loss, e + 1)
        print(f'[INFO] Epoch {e + 1}: CRPS (Training) = {train_loss}')
        return train_loss

    def validate_epoch(self, e: int, loader, conditions, tag='val'):
        tracker = WelfordStatisticsTracker()
        model = self.model
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(iter(loader)):
                predictors, observations = self.preprocess(batch, conditions, model, self.device, use_dequant=False)
                predictions = self.evaluate_model(predictors)
                loss = self.loss_func(predictions, observations)
                tracker.update(loss.item() / len(observations), weight=len(observations))
        val_loss = tracker.mean()
        if tag is not None:
            self.writer.add_scalar(f'loss/{tag}', val_loss, e + 1)
        label = {'test': 'Test', 'val': 'Validation', None: 'Final'}.get(tag, 'Undefined')
        print(f'[INFO] Epoch {e + 1}: CRPS ({label}) = {val_loss}')
        return val_loss

    def advance_scheduler(self, e: int, metric: float):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('lr', lr, e)
        print(f'[INFO] New learning rate: {lr}')
