#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
from typing import Callable, Union, Tuple, List, Dict
import time
from IPython.display import clear_output, display

import math
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import signal

from . import funcs
from .speckle_dataset import restore_u_ux_uy_translate_reshape_inplace

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# used for run_with_stop_handling
INTERRUPT = False
KEEPRUNNING = True
COMPUTE_LOOP_ARGS = ()
PATH_RESULTS = 'data/results/'
PATH_MODELS = 'data/models/'
###


# %% LOOP WITH INTERRUPT MANAGER
def compute_loop():
    global STILLRUNNING
    STILLRUNNING = True
    func_loop = COMPUTE_LOOP_ARGS
    end = False
    while KEEPRUNNING and (not end):
        end = func_loop()
    STILLRUNNING = False


def handle_interrupt(signal, frame):
    # Perform actions or cleanup tasks when Ctrl+C is pressed
    print("\n\n\nInterrupt signal received.\n\n")
    global KEEPRUNNING, INTERRUPT
    KEEPRUNNING = False
    INTERRUPT = True


def run_with_stop_handling(func: Callable):
    global KEEPRUNNING, COMPUTE_LOOP_ARGS
    KEEPRUNNING = True
    COMPUTE_LOOP_ARGS = func
    signal.signal(signal.SIGINT, handle_interrupt)
    compute_loop()


# %%
ACTIVATION = 'ReLU'


def set_activation(activation: str):
    global ACTIVATION
    ACTIVATION = activation


class Custom_lr_scheduler():
    def __init__(self, optimizer, function, func_args=()):
        self.optimizer = optimizer
        self.function = function
        self.funcs_args = func_args
        self.last_epoch = 0
        self.__update_optimizer__()

    def __update_optimizer__(self):
        lr = self.function(self.last_epoch, *self.funcs_args)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self):
        self.last_epoch += 1
        self.__update_optimizer__()


activations = {
    'ReLU': nn.ReLU(inplace=True),
    'LReLU': nn.LeakyReLU(inplace=True),
    'Sigmoid': nn.Sigmoid(),
    'ELU': nn.ELU(inplace=True),
    'SiLU': nn.SiLU(inplace=True),
    'Mish': nn.Mish(inplace=True),
}

OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'asgd': torch.optim.ASGD,
    'lbfgs': torch.optim.LBFGS,
    'rmsprop': torch.optim.RMSprop,
    'rprop': torch.optim.Rprop,
}

SCHEDULERS = {
    'custom': Custom_lr_scheduler,
    'linear': torch.optim.lr_scheduler.LinearLR,
    'multiplicative': torch.optim.lr_scheduler.MultiplicativeLR,
    'lambda': torch.optim.lr_scheduler.LambdaLR,
    'step': torch.optim.lr_scheduler.StepLR,
    'multi-step': torch.optim.lr_scheduler.MultiStepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosine annealing': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cyclic triangular': torch.optim.lr_scheduler.CyclicLR,
    'cyclic triangular 2': torch.optim.lr_scheduler.CyclicLR,
    'cyclic exp range': torch.optim.lr_scheduler.CyclicLR,
    'one cycle cos': torch.optim.lr_scheduler.OneCycleLR,
    'one cycle linear': torch.optim.lr_scheduler.OneCycleLR,
    'cosine annealing warm restart':
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}


def get_optimizer(model, optimizer_info):
    name = list(optimizer_info.keys())[0].lower()
    if name not in OPTIMIZERS:
        funcs.print_error(f'optimizer {name} not defined yet')
        return None
    return (OPTIMIZERS[name](model.parameters(),
                             **list(optimizer_info.values())[0]))


def get_scheduler(optimizer, scheduler_info):
    key = list(scheduler_info.keys())[0]
    name = key.lower()
    if name not in SCHEDULERS:
        funcs.print_error(f'Scheduler {name} not defined yet')
        return
    return SCHEDULERS[name](optimizer, **scheduler_info[key])


def get_scheduler_curve(optimizer_info, scheduler_info, epochs):
    model = torch.nn.Linear(2, 1)
    optimizer = get_optimizer(model, optimizer_info)
    scheduler = get_scheduler(optimizer, scheduler_info)
    lrs = []
    for ii in range(epochs+1):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs
###


# %%
def release_gpu_cache():
    torch.cuda.empty_cache()


class NN_Trainer():
    """Container to train, update and save Neural Network models"""
    def __init__(
        self,
        model: nn.Module,
        dataloaders: DataLoader,
        mode: str,
        loss_fn: Callable,
        weight_scaling: bool = False,
        test_loop: bool = True,
    ):
        """Container used to train, update and save Neural Netwok models

        Args:
            model (nn.Module): Neural Network model
            dataloaders (DataLoader): used to load data
            mode (str): only "classifier" is currently implemented
            weight_scaling (bool): wheter to call weight_scaling during
                                   training
        """
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = None
        self.scheduler = None
        self.metrics = {'train': [], 'test': []}
        self.losses = {'train': [], 'test': []}
        self.confusions = {'train': [], 'test': []}
        self.compute_time = 0.
        self.lrs = []
        self.loss_fn = loss_fn
        self.set_mode(mode)
        self.weight_scaling = weight_scaling
        self.test_loop = test_loop

    def __len__(self):
        return len(self.lrs)

    def set_mode(self, mode):
        mode = mode.lower()
        self.mode = mode

    def set_scheduler(self, scheduler_info: dict):
        if self.optimizer is None:
            funcs.print_error('set optimizer before scheduler')

        scheduler = get_scheduler(self.optimizer, scheduler_info)
        if scheduler is None:
            funcs.print_error('Scheduler not set')
            return
        self.scheduler_info = scheduler_info
        self.scheduler = scheduler

    def set_optimizer(self, optimizer_info: dict):
        """Set the optimizer for the Neural Network training.

        Args:
            optimizer_info (dict): key 'name' provide the optimizer name
                                   must contains 'params' attribute (dict)
                                   which corresponds to the parameters
                                   of the optimizer
        """
        optimizer = get_optimizer(self.model, optimizer_info)
        if optimizer is None:
            funcs.print_error('Optimizer not set')
            return
        self.optimizer_info = optimizer_info
        self.optimizer = optimizer

    def plot_lr_scheduler(self, title, savename, epochs):
        lrs = get_scheduler_curve(self.optimizer_info, self.scheduler_info,
                                  epochs)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(range(epochs), lrs)
        ax.set_xlabel('epoch')
        ax.set_ylabel('learning rate')
        ax.set_title(title)
        fig.tight_layout()
        graph_tools.savefig(fig, savename)

    def save_results(self, filename: str):
        filename = os.path.join(PATH_RESULTS, filename.replace(' ', '_'))
        funcs.make_folder_from_file(filename)
        if self.mode == 'classifier':
            metric_label = 'Accuracy'
        else:
            metric_label = 'MSE'
        funcs.to_pickle(filename, {'metric label': metric_label,
                                   'metric': self.metrics,
                                   'loss': self.losses,
                                   'lr': self.lrs,
                                   'confusion': self.confusions,
                                   'compute time': self.compute_time})

    def load_results(self, filename: str):
        data = funcs.load_trainer_results(filename)
        if data is None:
            return
        self.metrics = data['metric']
        self.losses = data['loss']
        self.lrs = data['lr']
        self.confusions = data['confusion']
        self.compute_time = data['compute time']

    def save_model(self, filename: str):
        self.save_results(filename)
        filename = os.path.join(PATH_MODELS, filename.replace(' ', '_'))
        funcs.make_folder_from_file(filename)
        to_save = {
            'mode': self.mode,
            'model_state_dict': self.model.state_dict(),
            }
        if self.optimizer is not None:
            to_save['optimizer_info'] = self.optimizer_info
            to_save['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            # to_save['scheduler_state_dict'] = self.scheduler.state_dict()
            to_save['scheduler_info'] = self.scheduler_info
            # to_save['scheduler_state_dict'] = self.scheduler.state_dict()
            to_save['last_epoch'] = self.scheduler.last_epoch
        torch.save(to_save, filename)

    def load_model(self, filename: str):
        self.load_results(filename)
        filename = os.path.join(PATH_MODELS, filename.replace(' ', '_'))

        checkpoint = torch.load(filename)
        # print(checkpoint["model_state_dict"].keys())
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_info' in checkpoint:
            self.set_optimizer(checkpoint['optimizer_info'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.optimizer = None
            self.optimizer_info = None
        if 'scheduler_info' in checkpoint:
            self.set_scheduler(checkpoint['scheduler_info'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scheduler.last_epoch = checkpoint['last_epoch']-1
            self.scheduler.step()
        else:
            self.scheduler = None
            self.scheduler_info = None
        self.set_mode(checkpoint['mode'])

    def train_network(self, epochs: int, print_each: int):
        """Begin or continue to train the current model.

        Args:
            epochs (int): maximum number of epochs from the start of the
                          training.
        """
        if len(self.lrs) > epochs:
            print(f'current number of epochs ({len(self.lrs)}) >= {epochs}')
            return
        if self.optimizer is None:
            funcs.print_error('optimizer not defined')
            return
        loss_fn = self.loss_fn
        metric_frmt = '{:.3e}'
        metric_lbl = 'MSE'
        confusion = False

        size = len(self.dataloaders['train'].dataset)
        release_gpu_cache()
        t0 = time.time()
        n_epoch_0 = len(self.lrs)

        self.tmp = (n_epoch_0, loss_fn, metric_lbl, metric_frmt, confusion,
                    size, t0, epochs, self.weight_scaling, print_each)

        run_with_stop_handling(self.__train_network_iter__)
        tend = time.time()

        t = len(self.lrs)

        delattr(self, 'tmp')
        # clear_output(wait=True)
        print(f"\nDone! ({t-n_epoch_0} epochs in {funcs.chrono(tend-t0)})")

        self.compute_time += tend - t0
        release_gpu_cache()

    def __train_network_iter__(self):
        (_, loss_fn, metric_lbl, metric_frmt, confusion,
            size, __, epochs, weight_scaling, print_each) = self.tmp

        losses_train = self.losses['train']
        losses_test = self.losses['test']
        t = len(self.lrs)
        losses_tr_plot = losses_train[max(0, t-5): t]
        losses_ts_plot = losses_test[max(0, t-5): t]
        losses_tr_plot = ', '.join([metric_frmt.format(value)
                                    for value in losses_tr_plot])
        losses_ts_plot = ', '.join([metric_frmt.format(value)
                                    for value in losses_ts_plot])
        # iteration
        clear_output(wait=True)
        print(f'Epoch {len(self.lrs)+1} / {epochs}, '
              + 'lr {:.9f}'.format(self.optimizer.param_groups[0]['lr']))
        print(f'train {metric_lbl}:', losses_tr_plot)
        print(f'test {metric_lbl}:', losses_ts_plot)
        loss_tr = __train_loop__(self.dataloaders['train'], self.model,
                                 loss_fn, self.optimizer, self.weight_scaling,
                                 print_each)
        # if weight_scaling:
        #     for layer in self.model.seq:
        #         if hasattr(layer, 'weight_scaling'):
        #             layer.weight_scaling()
        #     for layer in self.model.fc:
        #         if hasattr(layer, 'weight_scaling'):
        #             layer.weight_scaling()
        if not KEEPRUNNING:
            return len(self.lrs) > epochs
        # metric_tr, loss_tr, cm_tr = __test_loop__(self.dataloaders['train'],
        #                                           self.model, update_correct,
        #                                           loss_fn, confusion)
        # if not KEEPRUNNING:
        #     return len(self.lrs) > epochs

        if self.test_loop:
            loss_ts, cm_ts = __test_loop__(
                self.dataloaders['test'],
                self.model,
                loss_fn, confusion
            )

        self.losses['train'].append(loss_tr)
        self.lrs.append(self.optimizer.param_groups[0]["lr"])
        if confusion:
            self.confusions['train'].append(cm_tr)
        if self.test_loop:
            self.losses['test'].append(loss_ts)
            if confusion:
                self.confusions['test'].append(cm_ts)

        # self.metrics['train'].append(metric_tr)
        # self.metrics['test'].append(metric_ts)

        if self.scheduler is not None:
            self.scheduler.step()

        if not KEEPRUNNING:
            return len(self.lrs) > epochs

        return len(self.lrs) >= epochs


class NN_Trainer_disp_strain():
    """Container to train, update and save Neural Network models"""
    def __init__(
        self,
        model_disp: nn.Module,
        model_strain: nn.Module,
        dataloaders: DataLoader,
        weight_scaling: bool = False,
        test_loop: bool = True,
    ):
        """Container used to train, update and save Neural Netwok models

        Args:
            model (nn.Module): Neural Network model
            dataloaders (DataLoader): used to load data
            mode (str): only "classifier" is currently implemented
            weight_scaling (bool): wheter to call weight_scaling during
                                   training
        """
        self.model_disp = model_disp
        self.model_strain = model_strain
        self.dataloaders = dataloaders
        self.optimizer = None
        self.scheduler = None
        self.metrics = {
            'disp:': {'train': [], 'test': []},
            'strain': {'train': [], 'test': []},
        }
        self.losses = {
            'disp': {'train': [], 'test': []},
            'strain': {'train': [], 'test': []},
        }
        self.compute_time = 0.
        self.lrs = []
        self.loss_fn = nn.MSELoss()
        self.weight_scaling = weight_scaling
        self.test_loop = test_loop

    def set_scheduler(self, scheduler_info: dict):
        if self.optimizer_strain is None:
            funcs.print_error('set optimizers before schedulers')

        scheduler = get_scheduler(self.optimizer_disp, scheduler_info)
        if scheduler is None:
            funcs.print_error('Scheduler disp not set')
            return
        self.scheduler_disp = scheduler

        scheduler = get_scheduler(self.optimizer_strain, scheduler_info)
        if scheduler is None:
            funcs.print_error('Scheduler strain not set')
            return
        self.scheduler_strain = scheduler

        self.scheduler_info = scheduler_info

    def set_optimizer(self, optimizer_info: dict):
        """Set the optimizer for the Neural Network training.

        Args:
            optimizer_info (dict): key 'name' provide the optimizer name
                                   must contains 'params' attribute (dict)
                                   which corresponds to the parameters
                                   of the optimizer
        """
        optimizer = get_optimizer(self.model_disp, optimizer_info)
        if optimizer is None:
            funcs.print_error('Optimizer disp not set')
            return
        self.optimizer_disp = optimizer

        optimizer = get_optimizer(self.model_strain, optimizer_info)
        if optimizer is None:
            funcs.print_error('Optimizer strain not set')
            return
        self.optimizer_strain = optimizer

        self.optimizer_info = optimizer_info

    def plot_lr_scheduler(self, title, savename, epochs):
        lrs = get_scheduler_curve(self.optimizer_info, self.scheduler_info,
                                  epochs)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(range(epochs), lrs)
        ax.set_xlabel('epoch')
        ax.set_ylabel('learning rate')
        ax.set_title(title)
        fig.tight_layout()
        graph_tools.savefig(fig, savename)

    def save_results(self, filename: str):
        filename = os.path.join(PATH_RESULTS, filename.replace(' ', '_'))
        funcs.make_folder_from_file(filename)
        funcs.to_pickle(filename, {'metric label': 'MSE',
                                   'metric': self.metrics,
                                   'loss': self.losses,
                                   'lr': self.lrs,
                                   'compute time': self.compute_time})

    def load_results(self, filename: str):
        data = funcs.load_trainer_results(filename)
        if data is None:
            return
        self.metrics = data['metric']
        self.losses = data['loss']
        self.lrs = data['lr']
        self.compute_time = data['compute time']

    def save_model(self, filename: str):
        self.save_results(filename)
        filename = os.path.join(PATH_MODELS, filename.replace(' ', '_'))
        funcs.make_folder_from_file(filename)
        to_save = {
            'model_state_dict': self.model_disp.state_dict(),
        }
        if self.optimizer_disp is not None:
            to_save['optimizer_info'] = self.optimizer_info
            to_save['optimizer_state_dict'] = self.optimizer_disp.state_dict()
        if self.scheduler_disp is not None:
            # to_save['scheduler_state_dict'] = self.scheduler.state_dict()
            to_save['scheduler_info'] = self.scheduler_info
            to_save['last_epoch'] = self.scheduler_disp.last_epoch
        torch.save(to_save, filename + '-disp')

        to_save = {
            'model_state_dict': self.model_strain.state_dict(),
        }
        if self.optimizer_strain is not None:
            to_save['optimizer_info'] = self.optimizer_info
            to_save['optimizer_state_dict'] =\
                self.optimizer_strain.state_dict()
        if self.scheduler_strain is not None:
            # to_save['scheduler_state_dict'] = self.scheduler.state_dict()
            to_save['scheduler_info'] = self.scheduler_info
            to_save['last_epoch'] = self.scheduler_strain.last_epoch

        torch.save(to_save, filename + '-strain')

    def load_model(self, filename: str):
        self.load_results(filename)
        filename = os.path.join(PATH_MODELS, filename.replace(' ', '_'))

        checkpoint_strain = torch.load(filename + '-strain')
        self.model_strain.load_state_dict(
            checkpoint_strain.pop('model_state_dict')
        )

        checkpoint_disp = torch.load(filename + '-disp')
        self.model_disp.load_state_dict(
            checkpoint_disp.pop('model_state_dict')
        )

        if 'optimizer_info' in checkpoint_disp:
            self.set_optimizer(checkpoint_disp['optimizer_info'])
            self.optimizer_disp.load_state_dict(
                checkpoint_disp['optimizer_state_dict']
            )
            self.optimizer_strain.load_state_dict(
                checkpoint_strain['optimizer_state_dict']
            )
        else:
            self.optimizer = None
            self.optimizer_info = None
        if 'scheduler_info' in checkpoint_disp:
            self.set_scheduler(checkpoint_disp['scheduler_info'])
            self.scheduler_disp.last_epoch = checkpoint_disp['last_epoch']-1
            self.scheduler_disp.step()
            self.scheduler_strain.last_epoch = checkpoint_disp['last_epoch']-1
            self.scheduler_strain.step()
        else:
            self.scheduler_disp = None
            self.scheduler_strain = None
            self.scheduler_info = None

    def train_network(self, epochs: int):
        """Begin or continue to train the current model.

        Args:
            epochs (int): maximum number of epochs from the start of the
                          training.
        """
        if len(self.lrs) > epochs:
            print(f'current number of epochs ({len(self.lrs)}) >= {epochs}')
            return
        if self.optimizer_disp is None:
            funcs.print_error('optimizer disp not defined')
            return
        if self.optimizer_strain is None:
            funcs.print_error('optimizer strain not defined')
            return
        loss_fn = self.loss_fn
        metric_frmt = '{:.3f}'
        metric_lbl = 'MSE'

        size = len(self.dataloaders['train'].dataset)
        release_gpu_cache()
        t0 = time.time()
        n_epoch_0 = len(self.lrs)

        self.tmp = (n_epoch_0, loss_fn, metric_lbl, metric_frmt,
                    size, t0, epochs, self.weight_scaling)

        run_with_stop_handling(self.__train_network_iter__)
        tend = time.time()

        t = len(self.lrs)

        delattr(self, 'tmp')
        # clear_output(wait=True)
        print(f"\nDone! ({t-n_epoch_0} epochs in {funcs.chrono(tend-t0)})")

        self.compute_time += tend - t0
        release_gpu_cache()

    def __get_str_losses__(self, metric_frmt, which: str):
        losses = self.losses[which]
        losses_train = losses['train']
        losses_test = losses['test']
        t = len(self.lrs)
        losses_tr_plot = losses_train[max(0, t-5): t]
        losses_ts_plot = losses_test[max(0, t-5): t]
        losses_tr_plot = ', '.join([metric_frmt.format(value)
                                    for value in losses_tr_plot])
        losses_ts_plot = ', '.join([metric_frmt.format(value)
                                    for value in losses_ts_plot])
        return losses_tr_plot, losses_ts_plot

    def __train_network_iter__(self):
        (_, loss_fn, metric_lbl, metric_frmt,
            size, __, epochs, weight_scaling) = self.tmp

        losses_plot_disp = self.__get_str_losses__(metric_frmt, 'disp')
        losses_plot_strain = self.__get_str_losses__(metric_frmt, 'strain')

        # iteration
        clear_output(wait=True)
        print(f'Epoch {len(self.lrs)+1} / {epochs}, '
              + 'lr {:.9f}'.format(self.optimizer_disp.param_groups[0]['lr']))
        print(f'Displacement:\ntrain {metric_lbl}:', losses_plot_disp[0])
        if self.test_loop:
            print(f'test {metric_lbl}:', losses_plot_disp[1])
        print(f'Strain:\ntrain {metric_lbl}:', losses_plot_strain[0])
        if self.test_loop:
            print(f'test {metric_lbl}:', losses_plot_strain[1])

        losses_tr = __train_loop_2__(
            self.dataloaders['train'], self.model_disp, self.model_strain,
            loss_fn, self.optimizer_disp, self.optimizer_strain,
            self.weight_scaling
        )
        if not KEEPRUNNING:
            return len(self.lrs) > epochs

        if self.test_loop:
            print('TRAIENR DISP STRAIN: test loop to implement')
            # loss_ts, cm_ts = __test_loop__(
            #     self.dataloaders['test'],
            #     self.model,
            #     loss_fn, False
            # )

        self.losses['disp']['train'].append(losses_tr[0])
        self.losses['strain']['train'].append(losses_tr[1])
        self.lrs.append(self.optimizer_disp.param_groups[0]["lr"])

        if self.test_loop:
            self.losses['disp']['test'].append(losses_ts[0])
            self.losses['strain']['test'].append(losses_ts[1])

        if self.scheduler_disp is not None:
            self.scheduler_disp.step()

        if self.scheduler_strain is not None:
            self.scheduler_strain.step()

        if not KEEPRUNNING:
            return len(self.lrs) > epochs

        return len(self.lrs) >= epochs


# updated from pytorch tutorials
def __train_loop_2__(
            dataloader, model_disp, model_strain, loss_fn,
            optimizer_disp, optimizer_strain, weight_scaling
        ):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode
    #  important for batch normalization and dropout layers
    model_disp.train(), model_strain.train()

    if hasattr(dataloader.dataset, 'init_dataset0'):
        dataloader.dataset.init_dataset0()
    ur, vr = dataloader.dataset.uvr
    ur, vr = ur.to(DEVICE), vr.to(DEVICE)
    loss_disp_avg, loss_strain_avg = 0., 0.
    n = 0
    release_gpu_cache()
    optimizer_disp.zero_grad()
    optimizer_strain.zero_grad()
    for batch, (X, y, resample_infos) in enumerate(dataloader):
        y = y.to(DEVICE)
        if torch.isnan(y).sum() > 0:
            print('Y CONTAINS NAN')

        # DISP
        # Compute prediction and loss
        pred = model_disp(X, ur, vr, resample_infos)

        if torch.isnan(pred).sum() > 0:
            print(f'PRED DISP CONTAINS NAN (batch {batch})')

        loss_disp = loss_fn(pred, y[:, [0, 3]])
        loss_disp_avg += loss_disp.item()

        # Backpropagation
        loss_disp.backward()

        optimizer_disp.step()
        if weight_scaling:
            model_disp.weight_scaling()
        optimizer_disp.zero_grad()

        release_gpu_cache()

        # STRAIN
        # Compute prediction and loss
        pred = model_strain(X, ur, vr, resample_infos)

        if torch.isnan(pred).sum() > 0:
            print(f'PRED STRAIN CONTAINS NAN (batch {batch})')
        y = y.to(DEVICE)

        loss_strain = loss_fn(pred, y[:, [1, 2, 4, 5]])
        loss_strain_avg += loss_strain.item()

        # Backpropagation
        loss_strain.backward()

        optimizer_strain.step()
        if weight_scaling:
            model_strain.weight_scaling()
        optimizer_strain.zero_grad()

        n += len(X)
        del X

        release_gpu_cache()

        if batch % 4 == 0:
            loss_disp, loss_strain = loss_disp.item(), loss_strain.item()
            message = f"loss: {loss_disp:>4f} {loss_strain:>4f}"\
                      f" [{n:>5d}/{size:>5d}]     "
            print(message, end='\r')
        del y
        if not KEEPRUNNING:
            break
    if not isinstance(loss_disp, float):
        loss_disp, loss_strain = loss_disp.item(), loss_strain.item()
    message = f"loss: {loss_disp:>4f} {loss_strain:>4f}"\
              f" [{size:>5d}/{size:>5d}]     "
    print(message)
    # optimizer_disp.zero_grad(), optimizer_strain.zero_grad()
    release_gpu_cache()
    return loss_disp_avg / num_batches, loss_strain_avg / num_batches


# updated from pytorch tutorials
def __train_loop__(
        dataloader,
        model,
        loss_fn,
        optimizer,
        weight_scaling,
        print_each
        ):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode
    #  important for batch normalization and dropout layers
    model.train()

    if hasattr(dataloader.dataset, 'init_dataset0'):
        dataloader.dataset.init_dataset0()
    ur, vr = dataloader.dataset.uvr
    ur, vr = ur.to(DEVICE), vr.to(DEVICE)
    release_gpu_cache()
    loss_avg, n, batch = 0., 0, 0
    for X, y, resample_infos in dataloader:
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X, ur, vr, resample_infos)
        n += len(X)
        del X

        if torch.isnan(pred).sum() > 0:
            print(f'PRED CONTAINS NAN (batch {batch})')
        y = y.to(DEVICE)
        if torch.isnan(y).sum() > 0:
            print('Y CONTAINS NAN')
        loss = loss_fn(pred, y)
        loss_avg += loss.item()

        # Backpropagation
        loss.backward()

        optimizer.step()
        if weight_scaling:
            model.weight_scaling()

        batch += 1
        if batch % print_each == 0:
            loss = loss.item()
            # message = f"loss: {loss:>4f}  [{n:>5d}/{size:>5d}]     "
            message = f"loss: {loss:>.3e} {loss_avg/batch:>.3e}"\
                      f" [{n:>5d}/{size:>5d}]     "
            print(message, end='\r')
        del y
        if not KEEPRUNNING:
            break
    # if not isinstance(loss, float):
    #     loss = loss.item()
    # message = f"loss: {loss:>4f}  [{size:>5d}/{size:>5d}]     "
    message = f"loss: {loss_avg/num_batches:>.3e}"\
              f" [{n:>5d}/{size:>5d}]     "

    print(message)
    optimizer.zero_grad()
    release_gpu_cache()
    return loss_avg / num_batches


# updated from pytorch tutorials
def __test_loop__(dataloader: DataLoader, model: nn.Module,
                  loss_fn: Callable, confusion: bool):
    # Set the model to evaluation mode
    # important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    # correct = 0

    if hasattr(dataloader.dataset, 'init_dataset0'):
        dataloader.dataset.init_dataset0()
    # Evaluating the model with torch.no_grad() ensures that no gradients are
    # computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage
    # for tensors with requires_grad=True
    if confusion:
        true_labels = torch.empty(size, dtype=int)
        pred_labels = torch.empty_like(true_labels)
    ur, vr = dataloader.dataset.uvr
    ur, vr = ur.to(DEVICE), vr.to(DEVICE)
    n = 0
    with torch.no_grad():
        # release_gpu_cache()
        for batch, (X, y, resample_infos) in enumerate(dataloader):
            pred = model(X, ur, vr, resample_infos)
            if y.device != DEVICE:
                y = y.to(DEVICE)
            test_loss += loss_fn(pred, y).item()
            # correct += update_correct(y, pred)
            if confusion:
                for i, n in enumerate(range(n, n+len(y))):
                    true_labels[n] = y[i].argmax().item()
                    pred_labels[n] = pred[i].argmax().item()
                    n += 1
            else:
                n += X.shape[0]
            if not KEEPRUNNING:
                return -1, -1, None
            if batch % 4 == 0:
                message = f"test: [{n:>5d}/{size:>5d}]      "
                print(message, end='\r')
    message = f"test: [{n:>5d}/{size:>5d}]     "
    print(message)
    test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n {label}: {(100*correct):>0.1f}%, Avg loss: "
    #       f"{test_loss:>8f} \n")
    # del X, y
    # release_gpu_cache()
    if confusion:
        cm = confusion_matrix(true_labels, pred_labels)
    else:
        cm = None
    return test_loss, cm

###


# %%
def center_tensor1d(t):
    t.add_(-t.mean(-1, keepdim=True))


def normalize_centered_tensor1d(t):
    factors = 1. / torch.sqrt((t*t).sum(-1, keepdim=True))
    t.mul_(factors)


def center_tensor2d(t):
    t.add_(-t.mean((-1, -2), keepdim=True))


def normalize_centered_tensor2d(t):
    factors = 1. / torch.sqrt((t*t).sum((-1, -2), keepdim=True))
    t.mul_(factors)


def center_tensor3d(t):
    t.add_(-t.mean((-1, -2, -3), keepdim=True))


def normalize_centered_tensor3d(t):
    factors = 1. / torch.sqrt((t*t).sum((-1, -2, -3), keepdim=True))
    t.mul_(factors)


def weight_scaling1d(weight):
    with torch.no_grad():
        center_tensor1d(weight)
        normalize_centered_tensor1d(weight)


def weight_scaling2d(weight):
    with torch.no_grad():
        center_tensor2d(weight)
        normalize_centered_tensor2d(weight)


def weight_scaling3d(weight):
    with torch.no_grad():
        # center_tensor3d(weight)
        normalize_centered_tensor3d(weight)

###


# %%
def get_norm2d(norm: dict, out_channels: int):
    if norm is None:
        norm = {'which': 'batch'}
    norm = {key: value for key, value in norm.items()}
    which = norm.pop('which').lower()
    if which == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif which == 'group':
        num_groups = norm.pop('num_groups')
        norm = nn.GroupNorm(num_groups, out_channels, **norm)
    elif which == 'instance':
        norm = nn.InstanceNorm2d(out_channels, **norm)
    elif which == 'identity':
        norm = nn.Identity()
    return norm


def init_he(layer):
    with torch.no_grad():
        # He, recommanded with (leaky) relu only (pytorch)
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in',
                                 nonlinearity='leaky_relu')
        if hasattr(layer, 'bias') and (layer.bias is not None):
            nn.init.constant_(layer.bias, 0.)


class Double_conv(nn.Module):
    def __init__(
                self,
                in_channels: int,
                out_channels: int,
                # SepConvInfos: dict,
            ):
        super().__init__()
        # infos = {key: value for key, value in SepConvInfos.items()}
        self.seq = nn.Sequential(
            # Conv2d
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            activations[ACTIVATION],
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            activations[ACTIVATION]
        )

    def weight_scaling(self):
        weight_scaling3d(self.seq[0].weight)
        weight_scaling3d(self.seq[2].weight)

    def HE_init(self):
        init_he(self.seq[0])
        init_he(self.seq[2])

    def forward(self, x):
        return self.seq(x)


class SepConv2dComb(nn.Module):
    def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int,
                padding: int,
                bias: bool,
                groups_channels: int = None,
                n_subconv: int = 1,
                norm: dict = None
            ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv1 = nn.Conv2d(
            1, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        if n_subconv > 1:
            layers = [self.conv1]
            for i in range(1, n_subconv):
                layers.append(
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding, bias=False,
                        groups=out_channels
                    )
                )
            self.conv1 = nn.Sequential(*layers)

        if groups_channels is None:
            groups_channels = out_channels

        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels*out_channels, out_channels, kernel_size=1,
                stride=1, padding=0, bias=bias, groups=groups_channels
            ),
            get_norm2d(norm, self.out_channels),
            activations[ACTIVATION]
        )

    def weight_scaling(self):
        if isinstance(self.conv1, nn.Sequential):
            for layer in self.conv1:
                weight_scaling3d(layer.weight)
        else:
            weight_scaling3d(self.conv1.weight)

        weight_scaling3d(self.seq[0].weight)

    def HE_init(self):
        if isinstance(self.conv1, nn.Sequential):
            for layer in self.conv1:
                init_he(layer)
        else:
            init_he(self.conv1)
        init_he(self.seq[0])

    def forward(self, x):
        out = torch.empty(
            (x.shape[0], self.in_channels*self.out_channels) + x.shape[-2:],
            dtype=torch.float, device=x.device
            )
        for i in torch.arange(self.in_channels):
            out[:, i::self.in_channels, :, :] = self.conv1(x[:, i:i+1])
        # out = self.conv1(
        #     x.view(-1, 1, x.shape[-2], x.shape[-1])
        # ).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return self.seq(out)


class BottleNeck(nn.Module):

    expansion = 2

    def __init__(
                self,
                in_channels: int,
                out_channels: int,
                stride: int = 1,
                norm: dict = None,
                groups: int = 1,
            ):
        super().__init__()
        self.out_channels = out_channels * BottleNeck.expansion
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            ),
            get_norm2d(norm, out_channels),
            activations[ACTIVATION],
            nn.Conv2d(
                out_channels, out_channels, stride=stride, kernel_size=3,
                padding=1, bias=False, groups=groups
            ),
            get_norm2d(norm, out_channels),
            activations[ACTIVATION],
            nn.Conv2d(
                out_channels, out_channels * BottleNeck.expansion,
                kernel_size=1, bias=False
            ),
            get_norm2d(norm, out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Identity()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * BottleNeck.expansion,
                    stride=stride, kernel_size=1, bias=False
                ),
                get_norm2d(norm, out_channels * BottleNeck.expansion)
            )
        self.activation = activations[ACTIVATION]

    def weight_scaling(self):
        if not isinstance(self.shortcut, nn.Identity):
            weight_scaling3d(self.shortcut[0].weight)
        weight_scaling3d(self.residual_function[0].weight)
        weight_scaling3d(self.residual_function[3].weight)
        weight_scaling3d(self.residual_function[6].weight)

    def HE_init(self):
        if not isinstance(self.shortcut, nn.Identity):
            init_he(self.shortcut[0])
        init_he(self.residual_function[0])
        init_he(self.residual_function[3])
        init_he(self.residual_function[6])

    def forward(self, x):
        return self.activation(self.residual_function(x) + self.shortcut(x))


class DICNet(nn.Module):

    def __init__(
                self,
                in_channel: int,
                out_channel: int,
                upsample: str = 'bilinear',
                block: dict = None,
                norm: str = None,
                weight_scaling: bool = False,
                n_channels_0: int = None,
                num_block: List[int] = [1, 5, 6, 8, 5, 3]
            ):
        super().__init__()

        if block is None:
            block = {'which': 'bottleneck'}
        which = block['which'].lower()
        block_kwargs = {
            key: value for key, value in block.items()
            if key != 'which'
        }
        if which == 'bottleneck':
            block = BottleNeck
        else:
            print(f'DICNet ERROR: block {block} is unknown')
            return
        block = block, block_kwargs

        if len(num_block) != 6:
            print('DICNet ERROR: len(num_block) should be 6, '
                  f'but {len(num_block)} is given.')
            return

        self.b_weight_scaling = weight_scaling

        if n_channels_0 is None:
            n_channels_0 = 64

        self.in_channels = n_channels_0

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channel, self.in_channels, kernel_size=3,
                stride=1, padding=1, bias=False
            ),
            get_norm2d(norm, self.in_channels),
            activations[ACTIVATION]
        )

        N1 = 64
        N2 = 128
        N3 = 128
        N4 = 256
        N5 = 256
        N6 = 512
        self.conv1_x = self._make_layer(block, N1, num_block[0], 2, norm)
        self.conv2_x = self._make_layer(block, N2, num_block[1], 1, norm)
        self.conv3_x = self._make_layer(block, N3, num_block[2], 2, norm)
        self.conv4_x = self._make_layer(block, N4, num_block[3], 2, norm)
        self.conv5_x = self._make_layer(block, N5, num_block[4], 2, norm)
        self.conv6_x = self._make_layer(block, N6, num_block[5], 2, norm)

        self.upsample = nn.Upsample(
            scale_factor=2, mode=upsample, align_corners=True
        )
        M4 = 512
        M3 = 256
        M2 = 128
        M1 = 64
        expansion = BottleNeck.expansion
        self.dconv_up4 = Double_conv(expansion*(N6 + N5), M4)
        self.dconv_up3 = Double_conv(expansion*N4 + M4, M3)
        self.dconv_up2 = Double_conv(expansion*N3 + M3, M2)
        self.dconv_up1 = Double_conv(expansion*N2 + M2, M1)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(n_channels_0+M1, n_channels_0, 3, padding=1),
            get_norm2d(norm, n_channels_0),
            activations[ACTIVATION],
            nn.Conv2d(n_channels_0, out_channel, 1)
        )
        self.to(DEVICE)
        self.weight_scaling()

    def _make_layer(self, block, out_channels, num_blocks, stride, norm):
        block, kwargs = block
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_channels, out_channels, stride, norm, **kwargs)
            )
            self.in_channels = layers[-1].out_channels

        return nn.Sequential(*layers)

    def weight_scaling(self):
        if not self.b_weight_scaling:
            return

        if isinstance(self.conv1, nn.Sequential):
            weight_scaling3d(self.conv1[0].data.weight)
        else:
            self.conv1.weight_scaling()

        for layer in self.conv1_x:
            layer.weight_scaling()

        for layer in self.conv2_x:
            layer.weight_scaling()

        for layer in self.conv3_x:
            layer.weight_scaling()

        for layer in self.conv4_x:
            layer.weight_scaling()

        for layer in self.conv5_x:
            layer.weight_scaling()

        for layer in self.conv6_x:
            layer.weight_scaling()

        self.dconv_up4.weight_scaling()
        self.dconv_up3.weight_scaling()
        self.dconv_up2.weight_scaling()
        self.dconv_up1.weight_scaling()
        weight_scaling3d(self.dconv_last[0].weight)
        weight_scaling3d(self.dconv_last[3].weight)

    def HE_init(self):
        if isinstance(self.conv1, nn.Sequential):
            init_he(self.conv1[0])
        else:
            self.conv1.HE_init()

        for layer in self.conv1_x:
            layer.HE_init()

        for layer in self.conv2_x:
            layer.HE_init()

        for layer in self.conv3_x:
            layer.HE_init()

        for layer in self.conv4_x:
            layer.HE_init()

        for layer in self.conv5_x:
            layer.HE_init()

        for layer in self.conv6_x:
            layer.HE_init()

        self.dconv_up4.HE_init()
        self.dconv_up3.HE_init()
        self.dconv_up2.HE_init()
        self.dconv_up1.HE_init()
        init_he(self.dconv_last[0])
        init_he(self.dconv_last[3])

    def forward(self, x, ur, vr, resample_infos):
        # normalisation
        x = x.to(DEVICE)
        for xi in x:
            for xii in xi:
                xii -= xii.mean()
                xii /= xii.std()

        conv1 = self.conv1(x)  # [batch_size, 64, 128, 128]
        temp = self.conv1_x(conv1)  # [batch_size, 128, 64, 64]
        conv2 = self.conv2_x(temp)  # 2[batch_size, 256, 64, 64]
        conv3 = self.conv3_x(conv2)  # 2[batch_size, 256, 32, 32]
        conv4 = self.conv4_x(conv3)  # 2[batch_size, 512, 16, 16]
        conv5 = self.conv5_x(conv4)  # 2[batch_size, 512, 8, 8]
        bottle = self.conv6_x(conv5)  # 2[batch_size, 1024, 4, 4]

        x = self.upsample(bottle)  # [batch_size, 1024, 8, 8]

        x = torch.cat([x, conv5], dim=1)  # [batch_size, 1024+512, 8, 8]

        x = self.dconv_up4(x)  # [batch_size, 512, 8, 8]
        x = self.upsample(x)  # [batch_size, 512, 16, 16]

        x = torch.cat([x, conv4], dim=1)  # [batch_size, 512 + 512, 16, 16]

        x = self.dconv_up3(x)  # [batch_size, 256, 16, 16]
        x = self.upsample(x)  # [batch_size, 256, 32, 32]

        x = torch.cat([x, conv3], dim=1)  # [batch_size, 256 + 256, 32, 32]

        x = self.dconv_up2(x)  # [batch_size, 128, 32, 32]
        x = self.upsample(x)  # [batch_size, 128, 64, 64]

        x = torch.cat([x, conv2], dim=1)  # [batch_size, 128+256, 64, 64]

        x = self.dconv_up1(x)  # [batch_size, 64, 64, 64]
        x = self.upsample(x)  # [batch_size, 64, 128, 128]

        x = torch.cat([x, conv1], dim=1)  # [batch_size, 64+64, 128, 128]
        out = self.dconv_last(x)  # [batch_size, 2, 128, 128]

        resample_infos = resample_infos.to(DEVICE)
        for i in torch.arange(len(x)):
            restore_u_ux_uy_translate_reshape_inplace(
                out[i], ur, vr,
                *resample_infos[i]
            )
        return out


###


# %% END OF FILE
###
