#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import copy
import pickle

from typing import Callable, Union, Tuple, List, Dict
from IPython.display import clear_output, display

import pandas as pd
import signal
from sklearn.metrics import confusion_matrix

import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models
from torchviz import make_dot

from torch.nn import BatchNorm1d, BatchNorm2d, LocalResponseNorm,\
                     MaxPool1d, MaxPool2d, Dropout

from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

from common_vars import PATH_MODELS, PATH_RESULTS
import funcs

BILINEAR = transforms.InterpolationMode.BILINEAR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANNOTATION_FILENAME = 'data/annotation_file'
LABEL_IND_TO_STR_FILENAME = 'data/label_ind_to_str.pickle'

BATCH_SIZE = 32
BATCH_SIZE = 8

# used for run_with_stop_handling
KEEPRUNNING = True
COMPUTE_LOOP_ARGS = ()

print('Torch device:', DEVICE)
if str(DEVICE) == 'cuda':
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


# %%
def initialize_reproducibility():
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)


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


OPTIMIZERS = {'sgd': torch.optim.SGD,
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
    need_closure = str(OPTIMIZERS[name]).split(' ')[0] in ['LBFGS']
    return (OPTIMIZERS[name](model.parameters(),
                             **list(optimizer_info.values())[0]),
            need_closure)


def get_scheduler(optimizer, scheduler_info):
    key = list(scheduler_info.keys())[0]
    name = key.lower()
    if name not in SCHEDULERS:
        funcs.print_error(f'Scheduler {name} not defined yet')
        return
    return SCHEDULERS[name](optimizer, **scheduler_info[key])


def get_scheduler_curve(optimizer_info, scheduler_info, epochs):
    model = torch.nn.Linear(2, 1)
    optimizer, _ = get_optimizer(model, optimizer_info)
    scheduler = get_scheduler(optimizer, scheduler_info)
    lrs = []
    for ii in range(epochs+1):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


def release_gpu_cache():
    torch.cuda.empty_cache()


def print_model_structure(model, name: str):
    print(f'Model "{name}" structure: {model}\n\n')

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()}")
        # | Values : {param[:2]} \n")


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
    global KEEPRUNNING
    KEEPRUNNING = False


def run_with_stop_handling(func: Callable):
    global KEEPRUNNING, COMPUTE_LOOP_ARGS
    KEEPRUNNING = True
    COMPUTE_LOOP_ARGS = func
    signal.signal(signal.SIGINT, handle_interrupt)
    compute_loop()


# %%


# updated from pytorch tutorials
def __train_loop__(dataloader, model, loss_fn, optimizer, need_closure):
    size = len(dataloader.dataset)
    # Set the model to training mode
    #  important for batch normalization and dropout layers
    model.train()

    n = 0
    # release_gpu_cache()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X)
        n += len(X)
        del X
        if y.device != DEVICE:
            y = y.to(DEVICE)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        if need_closure:
            def closure():
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            optimizer.step()

        if batch % 4 == 0:
            loss = loss.item()
            message = f"loss: {loss:>4f}  [{n:>5d}/{size:>5d}]"
            print(message, end='\r')
        del y
        if not KEEPRUNNING:
            break
    if not isinstance(loss, float):
        loss = loss.item()
    message = f"loss: {loss:>4f}  [{size:>5d}/{size:>5d}]"
    print(message)
    optimizer.zero_grad()
    release_gpu_cache()


# updated from pytorch tutorials
def __test_loop__(dataloader: DataLoader, model: nn.Module,
                  update_correct: Callable, loss_fn: Callable,
                  confusion: bool):
    # Set the model to evaluation mode
    # important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are
    # computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage
    # for tensors with requires_grad=True
    if confusion:
        true_labels = torch.empty(size, dtype=int)
        pred_labels = torch.empty_like(true_labels)
        n = 0
    with torch.no_grad():
        # release_gpu_cache()
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            if y.device != DEVICE:
                y = y.to(DEVICE)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y.argmax(1)).sum().item()
            correct += update_correct(y, pred)
            if confusion:
                for i, n in enumerate(range(n, n+len(y))):
                    true_labels[n] = y[i].argmax().item()
                    pred_labels[n] = pred[i].argmax().item()
                n += 1
            if not KEEPRUNNING:
                return -1, -1, None
            if batch % 4 == 0:
                message = f"test: [{n:>5d}/{size:>5d}]"
                print(message, end='\r')
    message = f"test: [{n:>5d}/{size:>5d}]"
    print(message)
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n {label}: {(100*correct):>0.1f}%, Avg loss: "
    #       f"{test_loss:>8f} \n")
    # del X, y
    # release_gpu_cache()
    if confusion:
        cm = confusion_matrix(true_labels, pred_labels)
    else:
        cm = None
    return correct, test_loss, cm


def __prepare_image__(image: tensor, target_shape: tuple) -> tensor:
    """Resize input tensor corresponding to image according to
       the desired shape, and apply a gaussian blur

    Args:
        image (tensor): input image
        target_shape (tuple): desired shape

    Returns:
        tensor: resized_image
    """
    image = transforms.functional.autocontrast(image.to(DEVICE))
    image = transforms.functional.equalize(image)
    ht, wt = target_shape
    d, h, w = image.shape

    scale = max(image.shape) / max(target_shape)
    kernel_size = int(5*scale+0.5)
    gaussianBlur = transforms.GaussianBlur(kernel_size + (kernel_size+1) % 2,
                                           scale)
    image = gaussianBlur(image)

    if h > w:
        resize = transforms.Resize((ht, int(ht*w/h+0.5)), antialias=True)
        resized = resize(image)
        # crop = transforms.CenterCrop((ht, wt))
        # cropped = crop(resized)
    else:
        resize = transforms.Resize((int(wt*h/w+0.5), wt), antialias=True)
        resized = resize(image)
        # crop = transforms.CenterCrop((ht, wt))
        # cropped = crop(resized)
    # gaussianBlur = transforms.GaussianBlur(5, 1.)
    return resized


def base_transform(target_shape: tuple):
    return transforms.Compose([
                transforms.CenterCrop(target_shape),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])


def data_augmentation_transform(target_shape):
    transform = transforms.Compose([
                        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                               saturation=0.2, hue=0.1),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomRotation(degrees=30),
                        transforms.RandomAffine(degrees=30,
                                                translate=(0.2,)*2,
                                                scale=(0.9, 1.1),
                                                shear=None,
                                                interpolation=BILINEAR),
                        transforms.CenterCrop(size=target_shape),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                        ])
    return transform


class CustomImageDataset(Dataset):
    """Class to get images and labels
    An autoencoder mode is also present, for which __getitem__ return the image
    and a copy of it.
    """
    def __init__(self, which: str, img_dir: str, transform: Callable = None,
                 target_transform: Callable = None):
        """Initialisation of the class CustomImageDataset

        Args:
            which (str): train / test / validation / ...
            img_dir (str): path the the images
            transform (Callable, optional): function to transform images.
                                            Defaults to None.
            target_transform (Callable, optional): function to transform target
                                                   Defaults to None.
        """
        self.img_labels = pd.read_pickle(ANNOTATION_FILENAME + '_'
                                         + which + ".pickle")
        self.label_ind_to_str = pd.read_pickle(LABEL_IND_TO_STR_FILENAME)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.n_classes = self.img_labels.iloc[:, 1].max() + 1
        self.autoencoder_moder = False
        print('Nombre de classes dans le dataset:', self.n_classes)
        self.__getitem_current__ = self.__getitem_0__

        # memory_allocate = torch.cuda.memory_allocated() * (1e-6 / 8)
        # print('GPU memory (MB):', int(memory_allocate))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.__getitem_current__(idx)

    def __getitem_0__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torch.load(img_path, map_location=DEVICE).float()
        image /= image.max()
        if self.transform:
            image = self.transform(image)
        y = self.img_labels.iloc[idx, 1]
        label = torch.zeros(self.n_classes, dtype=torch.float,
                            device=DEVICE).scatter_(
                                0,
                                torch.tensor(y, device=DEVICE),
                                value=1)
        return image, label

    def __getitem_autoencodeur__(self, idx):
        image, label = self.__getitem_0__(idx)
        return image, image.clone()

    def set_autoencoder_mode(self, mode: bool):
        self.autoencoder_moder = mode
        if mode:
            self.__getitem_current__ = self.__getitem_autoencodeur__
            return
        self.__getitem_current__ = self.__getitem_0__

    def get_label_str(self, i_label):
        return self.label_ind_to_str[i_label]


def __init_layer__(which: str, params: Union[str, dict]):
    if which == 'activation':
        return activations[params]
    else:
        return eval(which)(**params)


def sequential_initialization(sequences: Union[List[Tuple[str, dict]],
                                               nn.Sequential]):
    if isinstance(sequences, nn.Sequential):
        return sequences
    layers = []
    for which, params in sequences:
        layers.append(__init_layer__(which, params))
    return nn.Sequential(*layers)


class NeuralNetwork(nn.Module):
    def __init__(self,
                 layers: List[Dict[str, object]] = None,
                 fc: List[Dict[str, object]] = None,
                 filename: str = None):
        super().__init__()
        self.breakpoint = None
        if filename is None:
            self.seq = sequential_initialization(layers)
            self.fc = sequential_initialization(fc)
        else:
            self.load_from_file(filename)
        release_gpu_cache()
        self.to(DEVICE)

    def forward(self, x):
        # print('x:', x.shape)
        if x.device != DEVICE:
            x = x.to(DEVICE)
        return self.fc(self.seq(x))

    def init_last_dense_classifier(self, n_elem_per_class):
        init_final_linear_classifier_layer(self.fc[-1], n_elem_per_class)

    def save_to_file(self, filename):
        filename = filename.replace(' ', '_')
        if '.pth' != filename[:-3]:
            filename += '.pth'
        torch.save(self.state_dict(), os.path.join(PATH_MODELS, filename))

    def load_from_file(self, filename):
        filename = filename.replace(' ', '_')
        if '.pth' != filename[:-3]:
            filename += '.pth'
        self.load_state_dict(torch.load(os.path.join(PATH_MODELS, filename)))

    def set_breakpoint(self):
        self.breakpoint = copy.deepcopy(self.state_dict())

    def reload_breakpoint(self):
        if self.breakpoint is not None:
            self.load_state_dict(self.breakpoint)
        else:
            print('\n!!! Neural Network : no available breakpoint !!!\n')


class NN_Trainer():
    """Container to train, update and save Neural Network models"""
    def __init__(self, model: nn.Module, dataloaders: DataLoader, mode: str):
        """Container used to train, update and save Neural Netwok models

        Args:
            model (nn.Module): Neural Network model
            dataloaders (DataLoader): used to load data
            mode (str): _description_
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
        self.set_mode(mode)

    def set_mode(self, mode):
        mode = mode.lower()
        self.mode = mode
        if mode == 'autoencoder':
            self.dataloaders['train'].dataset.set_autoencoder_mode(True)
            self.dataloaders['test'].dataset.set_autoencoder_mode(True)
        else:
            self.dataloaders['train'].dataset.set_autoencoder_mode(False)
            self.dataloaders['test'].dataset.set_autoencoder_mode(False)

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
        self.optimizer, self.need_closure = optimizer

    def plot_lr_scheduler(self, title, savename, epochs):
        lrs = get_scheduler_curve(self.optimizer_info, self.scheduler_info,
                                  epochs)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(range(epochs), lrs)
        ax.set_xlabel('epoch')
        ax.set_ylabel('learning rate')
        ax.set_title(title)
        fig.tight_layout()
        graph_tools.savefig(fig, PATHS_PRINT['essais'] + savename)

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
        data = load_trainer_results(filename)
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
            to_save['optimizer_info']: self.optimizer_info
            to_save['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            # to_save['scheduler_state_dict'] = self.scheduler.state_dict()
            to_save['scheduler_info'] = self.scheduler_info
            to_save['last_epoch'] = self.scheduler.last_epoch
        torch.save(to_save, filename)

    def load_model(self, filename: str):
        self.load_results(filename)
        filename = os.path.join(PATH_MODELS, filename.replace(' ', '_'))

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimiser_info' in checkpoint:
            self.set_optimizer(checkpoint['optimizer_info'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.optimizer = None
            self.optimizer_info = None
        if 'scheduler_info' in checkpoint:
            self.set_scheduler(checkpoint['scheduler_info'])
            self.scheduler.last_epoch = checkpoint['last_epoch']
        else:
            self.scheduler = None
            self.scheduler_info = None
        self.set_mode(checkpoint['mode'])

    def train_network(self, epochs: int, lr_mode: str = 'scheduler',
                      init_lr: float = 1e-3):
        """Begin or continue to train the current model.

        Args:
            epochs (int): maximum number of epochs from the start of the
                          training.
            lr_mode (str, optional): scheduler or LM. Defaults to 'scheduler'.
            init_lr (float, optional): initial learning rate value used for
                                       the LM loop if lr_mode is set to LM.
                                       Defaults to 1e-3.
        """
        if len(self.lrs) > epochs:
            print(f'current number of epochs ({len(self.lrs)}) >= {epochs}')
            return
        if self.optimizer is None:
            funcs.print_error('optimizer not defined')
            return
        if self.mode == 'classifier':
            loss_fn = nn.CrossEntropyLoss()
            metric_frmt = '{:.2%}'
            metric_lbl = 'Accuracy'
            confusion = True
            confusion_train = self.confusions['train']
            confusion_test = self.confusions['test']

            def update_correct(y, pred):
                return (pred.argmax(1) == y.argmax(1)).sum().item()
        else:
            loss_fn = nn.MSELoss()
            metric_frmt = '{:.3f}'
            metric_lbl = 'MSE'
            confusion = False

            def update_correct(y, pred):
                return ((pred-X)**2).sum().item()

        size = len(self.dataloaders['train'].dataset)
        release_gpu_cache()
        t0 = time.time()
        n_epoch_0 = len(self.lrs)-1

        if n_epoch_0 < 0:
            n_epoch_0 = 0
            (metric_tr,
             loss_tr,
             cm_tr) = __test_loop__(self.dataloaders['train'],
                                    self.model, update_correct,
                                    loss_fn, confusion)

            (metric_ts,
             loss_ts,
             cm_ts) = __test_loop__(self.dataloaders['test'],
                                    self.model, update_correct,
                                    loss_fn, confusion)
            self.losses['train'].append(loss_tr)
            self.losses['test'].append(loss_ts)
            self.metrics['train'].append(metric_tr)
            self.metrics['test'].append(metric_ts)
            self.lrs.append(self.optimizer.param_groups[0]["lr"])
            if confusion:
                self.confusions['train'].append(cm_tr)
                self.confusions['test'].append(cm_ts)

        self.tmp = (n_epoch_0, loss_fn, metric_lbl, metric_frmt, confusion,
                    size, update_correct, t0, epochs)

        if lr_mode == 'scheduler':
            run_with_stop_handling(self.__train_network_iter__)
        elif lr_mode == 'LM':
            self.model.set_breakpoint()
            if len(self.lrs) > 1:
                self.current_lr = self.lrs[-1]
            else:
                self.current_lr = init_lr
            run_with_stop_handling(self.__train_network_iter_LM__)
        tend = time.time()

        t = len(self.lrs)-1

        delattr(self, 'tmp')
        # clear_output(wait=True)
        print(f"\nDone! ({t-n_epoch_0} epochs in {funcs.chrono(tend-t0)})")

        self.compute_time += tend - t0
        release_gpu_cache()

    def __train_network_iter__(self):
        (_, loss_fn, metric_lbl, metric_frmt, confusion,
            size, update_correct, __, epochs) = self.tmp

        metrics_train = self.metrics['train']
        metrics_test = self.metrics['test']
        t = len(self.lrs)
        metric_tr_plot = metrics_train[max(0, t-5): t]
        metric_ts_plot = metrics_test[max(0, t-5): t]
        metric_tr_plot = ', '.join([metric_frmt.format(value)
                                    for value in metric_tr_plot])
        metric_ts_plot = ', '.join([metric_frmt.format(value)
                                    for value in metric_ts_plot])
        # iteration
        clear_output(wait=True)
        print(f'Epoch {len(self.lrs)} / {epochs}, '
              + 'lr {:.9f}'.format(self.optimizer.param_groups[0]['lr']))
        print(f'train {metric_lbl}:', metric_tr_plot)
        print(f'test {metric_lbl}:', metric_ts_plot)
        __train_loop__(self.dataloaders['train'], self.model, loss_fn,
                       self.optimizer, self.need_closure)

        if not KEEPRUNNING:
            return len(self.lrs) > epochs
        metric_tr, loss_tr, cm_tr = __test_loop__(self.dataloaders['train'],
                                                  self.model, update_correct,
                                                  loss_fn, confusion)
        if not KEEPRUNNING:
            return len(self.lrs) > epochs
        metric_ts, loss_ts, cm_ts = __test_loop__(self.dataloaders['test'],
                                                  self.model, update_correct,
                                                  loss_fn, confusion)
        if not KEEPRUNNING:
            return len(self.lrs) > epochs

        self.losses['train'].append(loss_tr)
        self.losses['test'].append(loss_ts)
        self.metrics['train'].append(metric_tr)
        self.metrics['test'].append(metric_ts)
        self.lrs.append(self.optimizer.param_groups[0]["lr"])
        if confusion:
            self.confusions['train'].append(cm_tr)
            self.confusions['test'].append(cm_ts)
        if self.scheduler is not None:
            self.scheduler.step()
        return len(self.lrs) > epochs

    # Levenberg-Marquardt like algorithm
    def __train_network_iter_LM__(self):
        (_, loss_fn, metric_lbl, metric_frmt, confusion,
            size, update_correct, __, epochs) = self.tmp

        metrics_train = self.metrics['train']
        metrics_test = self.metrics['test']
        t = len(self.lrs)
        metric_tr_plot = metrics_train[max(0, t-5): t]
        metric_ts_plot = metrics_test[max(0, t-5): t]
        metric_tr_plot = ', '.join([metric_frmt.format(value)
                                    for value in metric_tr_plot])
        metric_ts_plot = ', '.join([metric_frmt.format(value)
                                    for value in metric_ts_plot])

        # iteration
        while KEEPRUNNING:
            clear_output(wait=True)
            print(f'Epoch {len(self.lrs)} / {epochs}, '
                  + 'lr {:.5f}'.format(self.optimizer.param_groups[0]['lr']))
            print(f'train {metric_lbl}:', metric_tr_plot)
            print(f'test {metric_lbl}:', metric_ts_plot)
            __train_loop__(self.dataloaders['train'], self.model, loss_fn,
                           self.optimizer, self.need_closure)
            if not KEEPRUNNING:
                break
            (metric_tr, loss_tr,
                cm_tr) = __test_loop__(self.dataloaders['train'],
                                       self.model, update_correct,
                                       loss_fn, confusion)
            if not KEEPRUNNING:
                break
            if loss_tr > self.losses['train'][-1]:
                self.model.reload_breakpoint()
                self.current_lr *= 0.8
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                continue
            break

        if not KEEPRUNNING:
            return len(self.lrs) > epochs
        self.model.set_breakpoint()

        metric_ts, loss_ts, cm_ts = __test_loop__(self.dataloaders['test'],
                                                  self.model, update_correct,
                                                  loss_fn, confusion)

        self.losses['train'].append(loss_tr)
        self.losses['test'].append(loss_ts)
        self.metrics['train'].append(metric_tr)
        self.metrics['test'].append(metric_ts)
        self.lrs.append(self.optimizer.param_groups[0]["lr"])
        if confusion:
            self.confusions['train'].append(cm_tr)
            self.confusions['test'].append(cm_ts)

        self.current_lr *= 1.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        return len(self.lrs) > epochs


def get_conv_resize(size_in, kernel, stride, padding):
    return 1 + int((size_in - kernel + 2*padding)/stride)


def init_dense_linear(layer):
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.constant_(layer.bias, 0.)


def init_final_linear_classifier_layer(layer: nn.Module, n_elem_per_class):
    """_summary_

    Args:
        layer (nn.Module): Linear layer
        n_elem_per_class (list / tensor / ...): number of elements per class
    """
    if isinstance(layer, Dense):
        layer = layer.seq[0]
    n_elem_per_class = torch.as_tensor(n_elem_per_class)
    frequencies = n_elem_per_class / n_elem_per_class.sum()
    with torch.no_grad():
        layer.bias[:] = torch.log(frequencies)
        # layer.bias[:] = frequencies[:]


def init_he(layer):
    with torch.no_grad():
        # He, recommanded with (leaky) relu only (pytorch)
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in',
                                 nonlinearity='leaky_relu')
        if hasattr(layer, 'bias') and (layer.bias is not None):
            nn.init.constant_(layer.bias, 0.)


# %%
def center_tensor2d(t):
    t.add_(-t.mean((-1, -2)).reshape(t.shape[:-2] + (1, 1)))


def normalize_centered_tensor2d(t):
    factors = 1. / torch.sqrt((t*t).sum((-1, -2)))
    t.mul_(factors.reshape(t.shape[:-2] + (1, 1)))


class GlbAvg2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.amean((-1, -2))


class GlbMax2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.amax((-1, -2))


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features, bias: bool = True,
                 BatchNorm: bool = False, activation: str = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        cl = nn.Linear(in_features, out_features, bias=bias)
        layer = [cl]
        if BatchNorm:
            layer.append(nn.BatchNorm1d(out_features))
        if activation is not None:
            layer.append(activations[activation])
        self.seq = nn.Sequential(*layer)
        if len(layer) > 1:
            init_he(cl)
        else:
            init_dense_linear(cl)

    def forward(self, x):
        return self.seq(x)

    def __repr__(self):
        # Customize the representation when printed
        representation = "Dense:\n"
        for i, layer in enumerate(self.seq):
            representation += f"  ({i}): {layer.__repr__()}\n"
        return representation


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: int = 1, padding: int = 0, groups: int = 1,
                 BatchNorm: bool = False, activation: str = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.BatchNorm = BatchNorm
        self.activation = activation
        # no bias for conv to avoid redondancy with BatchNorm
        cl = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                       stride=stride, padding=padding, groups=groups,
                       bias=not BatchNorm)
        init_he(cl)
        layer = [cl]
        if BatchNorm:
            layer.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layer.append(activations[activation])
        self.seq = nn.Sequential(*layer)

    def forward(self, x):
        return self.seq(x)

    def __repr__(self):
        # Customize the representation when printed
        return self.seq.__repr__()


class Conv2d_SepSpace(nn.Module):
    def Conv2d_SepSpace(in_channels: int, out_channels: int,
                        kernel_sizes: Tuple[int], strides: Tuple[int],
                        paddings: Tuple[int], BatchNorm: bool,
                        activation: Callable = None):
        """Currently design for first layer, i.e. in_channels = 3,
        out_channels = m * 3

        Args:
            in_channels (int): number of input_channels
            out_channels (int): number of channels / filters
            kernel_sizes (Tuple[int]): size of each 1D filter pair in the chain
            strides (Tuple[int]): stride for each 1D filter pair in the chain
            paddings (Tuple[int]): padding for each 1D filter pair in the chain
            BatchNorm (bool): wheter to add BatchNorm at the end or not
            activation (Callable, optional): activaiton fuction append at the end
                                            of the block. Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        groups = min(in_channels, out_channels)
        bias = not BatchNorm
        layer = []
        for kernel_size, stride, padding in zip(kernel_sizes, strides,
                                                paddings):
            cli = nn.Conv2d(in_channels, out_channels,
                            kernel_size=(kernel_size, 1),
                            stride=(stride, 1), padding=(padding, 0),
                            bias=bias, groups=groups)
            init_he(cli)
            clj = nn.Conv2d(out_channels, out_channels,
                            kernel_size=(1, kernel_size),
                            stride=(1, stride), padding=(0, padding),
                            bias=False, groups=groups)
            init_he(clj)
            layer += [cli, clj]
            in_channels = out_channels
        if BatchNorm:
            layer.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layer.append(activation)
        self.seq = nn.Sequential(*layer)

    def forward(self, x):
        return self.seq(x)


class ResidualUnit(nn.Module):
    def __init__(self, n_input: int, n_out: int, n_subspace: int,
                 C: int, activation: Callable, add_SEblock: bool):
        super().__init__()
        self.n_input = n_input
        self.n_out = n_out
        self.n_subspace = n_subspace
        self.C = C

        self.cl0 = nn.Conv2d(n_input, n_subspace, kernel_size=1,
                             stride=1, padding=0, bias=False)
        init_he(self.cl0)
        self.bn0 = nn.BatchNorm2d(n_subspace)

        self.cl1 = nn.Conv2d(n_subspace, n_subspace, kernel_size=3,
                             stride=1, padding=1,
                             bias=False, groups=self.C)
        init_he(self.cl1)
        self.bn1 = nn.BatchNorm2d(n_subspace)

        self.cl2 = nn.Conv2d(n_subspace, n_out, kernel_size=1,
                             stride=1, padding=0, bias=False)
        init_he(self.cl2)
        self.bn2 = nn.BatchNorm2d(n_out)

        if add_SEblock:
            self.SEblock = SEblock(n_out)
        self.add_SEblock = add_SEblock

        if n_input != n_out:
            self.conv_Id = True
            self.cl3 = nn.Conv2d(n_input, n_out, kernel_size=1,
                                 stride=1, padding=0, bias=False)
            self.cl3.weight.data.fill_(1.)
            # self.cl3.bias.data.fill_(0.)
            self.bn3 = nn.BatchNorm2d(n_out)
        else:
            self.conv_Id = False
        self.activation = activation

    def forward(self, x):
        # print('\n\nRU\nx:', x.shape)
        out = self.cl0(x)
        out = self.bn0(out)
        out = self.activation(out)
        out = self.cl1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.cl2(out)
        out = self.bn2(out)

        if self.add_SEblock:
            # print('out:', out.shape)
            out = out * self.SEblock(out)

        # skip connection
        # print('skip connection:')
        # print('out:', out.shape)
        if self.conv_Id:
            tmp = self.cl3(x)
            out = out + self.bn3(tmp)
        else:
            out = out + x
        out = self.activation(out)
        # print('out:', out.shape)

        return out


class SEblock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        dense0 = nn.Linear(n_channels, int(n_channels//16))
        dense2 = nn.Linear(int(n_channels//16), n_channels)
        init_he(dense0)
        init_he(dense2)
        self.seq = nn.Sequential(
            GlbAvg2d(),
            dense0,
            nn.BatchNorm1d(int(n_channels//16)),
            activations['SiLU'],
            dense2,
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x).reshape(x.shape[:-2] + (1, 1))


class ZNCC2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias: bool = True):
        super().init()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.pattern = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.sum2d = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            divisor_override=1,
        )
        self.size_ = kernel_size**2 if isinstance(kernel_size, int)\
            else kernel_size[0] * kernel_size[1]
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init_he(self.pattern)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1. / torch.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def center_norm_pattern(self):
        with torch.no_grad():
            center_tensor2d(self.pattern.weight)
            normalize_centered_tensor2d(self.pattern.weight)

    def forward(self, x):
        # Conv2d pattern
        Hx = self.pattern(x)
        sum_x2 = self.sum2d(x*x)
        sum_x = self.sum2d(x)
        out = Hx / (sum_x2 - (sum_x*sum_x)/self.size_)
        if self.bias


class myLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        # Here we used torch.nn.Parameter to set our weight and bias,
        # otherwise, it won’t train.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        _, y = input.shape
        if y != self.in_features:
            sys.exit(f'Wrong Input Features. Please use tensor with '
                     f'{self.in_features} Input Features')
        output = input @ self.weight.t() + self.bias
        return output


def AlexNet_based_architecture(n_out: int, activation: str,
                               separable_conv_begin: bool = False,
                               separable_conv2_begin: bool = False,
                               ResidualUnits: bool = False,
                               RU_params: dict = {'n_subspace': 128,
                                                  'C': 32,
                                                  'add_SEblock': False},
                               endPoolType: str = 'max',
                               n_elem_per_class: Tuple[int, ...] = None):
    activation = activations[activation]

    n_filters_in = 96
    n_filters_2 = 192
    # n_filters_2 = 256
    # n_filters_3 = 256

    # n_filters_in = 129
    # n_filters_2 = 256
    n_filters_3 = 192

    # shape: 224 x 224
    if separable_conv_begin:
        conv0 = Conv2d_SepSpace(3, n_filters_in, kernel_sizes=(7, 5),
                                strides=(4, 1), paddings=(0, 0),
                                activation=activation)
    else:
        conv0 = Conv2d_BN_Activation(3, n_filters_in, kernel_size=11, stride=4,
                                     padding=0, activation=activation)
    # shape: 54 x 54
    maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    lrn0 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.)

    # shape: 26 x 26
    if separable_conv2_begin:
        conv2 = Conv2d_SepSpace(n_filters_in, n_filters_2,
                                kernel_sizes=(3, 3, 3),
                                strides=(1, 1, 1), paddings=(0, 0, 1),
                                activation=activation)
    else:
        conv2 = Conv2d_BN_Activation(n_filters_in, n_filters_2, kernel_size=5,
                                     stride=1, padding=0,
                                     activation=activation)
    # shape: 22 x 22
    maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 10 x 10
    if ResidualUnits:
        conv4 = ResidualUnit(n_filters_2, n_filters_3, activation=activation,
                             **RU_params)
    else:
        conv4 = Conv2d_BN_Activation(n_filters_2, n_filters_3, kernel_size=3,
                                     stride=1, padding=1,
                                     activation=activation)

    # shape: 10 x 10
    if endPoolType.lower() == 'max':
        endPool5 = GlbMax2d()
    elif endPoolType.lower() == 'average':
        # endPool5 = nn.AvgPool2d(kernel_size=10, stride=1, padding=0)
        endPool5 = GlbAvg2d()

    n_subspace_end = 128
    n_channels_end = n_filters_3
    # shape: 1 x 1
    dense6 = Dense(in_features=n_channels_end, out_features=n_subspace_end,
                   bias=True, BatchNorm=False, activation=activation)
    init_he(dense6)
    dense7 = Dense(in_features=n_channels_end, out_features=n_subspace_end,
                   bias=True, BatchNorm=False, activation=activation)
    init_he(dense7)
    dense8 = Dense(in_features=n_subspace_end, out_features=n_out, bias=True,
                   BatchNorm=False, activation=False)
    init_dense_linear(dense8)
    if n_elem_per_class is None:
        n_elem_per_class = [1,] * n_out
    init_final_linear_classifier_layer(dens8, n_elem_per_class)

    fc = nn.Sequential(
        dense6,
        nn.Dropout(p=0.5, inplace=False),
        dense7,
        nn.Dropout(p=0.5, inplace=False),
        dense8
    )

    return nn.Sequential(
        conv0,
        maxPool1,
        lrn0,
        conv2,
        maxPool3,
        conv4,
        endPool5, nn.Flatten(),
        dense6, bn6, activation,
        nn.Dropout(p=0.5, inplace=False),
        dense7, bn7, activation,
        nn.Dropout(p=0.5, inplace=False),
        dense8
    )


def ResNeXt_pretrained(out_features):
    model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    transforms = ResNeXt50_32X4D_Weights.IMAGENET1K_V2.transforms
    # fix parameters
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = DenseStack(num_features, out_features=out_features, n_layers=3,
                          n_subspace=128, activation=activations['SiLU'],
                          dropout=0.5)
    return model, transforms

# %% END OF FILE
###
