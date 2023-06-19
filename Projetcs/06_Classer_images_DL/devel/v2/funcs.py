#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% IMPORT PACKAGES
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
import copy
import pickle
from typing import Callable, Union, Tuple
from IPython.display import clear_output, display

# import tkinter as tk
import signal

from sklearn import model_selection
from sklearn.metrics import confusion_matrix

import torch
from torch import nn, tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models
from torchviz import make_dot

from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

import matplotlib.pyplot as plt
import utilities_tools_and_graph as graph_tools

BILINEAR = transforms.InterpolationMode.BILINEAR

PATH_DATA = 'data/'
PATH_RESULTS = PATH_DATA + 'results/'
PATH_MODELS = PATH_DATA + 'models/'
PATHS_PRINT = {'explore': 'Figures/explore/',
               'essais': 'Figures/essais/'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANNOTATION_FILENAME = 'data/annotation_file'
LABEL_IND_TO_STR_FILENAME = 'data/label_ind_to_str.pickle'

BATCH_SIZE = 32
BATCH_SIZE = 8
# NUMWORKERS = 6

# used for run_with_stop_handling
KEEPRUNNING = True
COMPUTE_LOOP_ARGS = ()

print('Torch device:', DEVICE)
if str(DEVICE) == 'cuda':
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
###


# %% tools
def print_error(message: str):
    print(f'\n!!! ERROR: {message} !!!\n')


def make_folder(path_folder: str):
    """Create a folder given the path. Test is performed first to check if
    folder already exists or not.

    Args:
        path_folder (str): path to the folder
    """
    path_folder = str(path_folder)
    try:
        if os.path.isdir(path_folder):
            return
        os.makedirs(path_folder)
    except OSError:
        pass
    return


def to_pickle(filename, data):
    filename = filename.replace(' ', '_')
    savename = os.path.join(filename + '.pickle')
    if os.sep in savename:
        make_folder(savename[:savename.rindex(os.sep)])
    with open(savename, 'wb') as file:
        pickle.dump(data, file)


def load_pickle(filename):
    filename = filename.replace(' ', '_') + '.pickle'
    if not os.path.isfile(filename):
        print_error(f'load pickel :file "{filename}" not found')
        return None
    with open(os.path.join(filename), 'rb') as file:
        data = pickle.load(file)
    return data


def chrono(sec):
    hours = int(sec // 3600)
    sec -= 3600 * hours
    mins = int(sec // 60)
    return f'{hours:02}:{mins:02}:{sec-60*mins:05.2f}'


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


def compute_loop_popup():
    global STILLRUNNING
    window, func_loop = COMPUTE_LOOP_ARGS
    if not KEEPRUNNING:
        STILLRUNNING = False
        window.after(0, window.destroy())
        return
    end = func_loop()
    if not end:
        window.after(0, compute_loop)
    else:
        STILLRUNNING = False
        window.after(0, window.destroy())


def set_stop_running():
    print('\n\n\nRequiring stop running')
    window, _ = COMPUTE_LOOP_ARGS
    global KEEPRUNNING
    KEEPRUNNING = False
    window.close_button.config(text='Stop at next step')
    window.close_button.config(state='disable')


def center_window(window):
    # Get the screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calculate the x and y coordinates for centering the window
    x = (screen_width - window.winfo_reqwidth()) // 2
    y = (screen_height - window.winfo_reqheight()) // 2

    # Set the window's position
    window.geometry(f"+{int(x)}+{int(y)}")


POPUP_OPEN = False


def create_popup():
    global POPUP_OPEN
    if not POPUP_OPEN:
        # Create a tk window to run
        popup = tk.Tk()
        close_button = tk.Button(popup, text="Stop computation",
                                 command=set_stop_running)
        close_button.pack()
        popup.close_button = close_button
        # center_window(popup)

        POPUP_OPEN = True
    return popup


def run_with_stop_button(func: Callable):
    # # Get the screen width and height
    # screen_width = popup.winfo_screenwidth()
    # screen_height = popup.winfo_screenheight()
    # # Calculate the x and y coordinates for centering the popup
    # x = (screen_width - popup.winfo_reqwidth()) // 2
    # y = (screen_height - popup.winfo_reqheight()) // 2
    # # Set the popup's position
    # popup.geometry(f"+{x}+{y}")

    global KEEPRUNNING, COMPUTE_LOOP_ARGS, POPUP_OPEN, STILLRUNNING
    POPUP_OPEN = False
    KEEPRUNNING = True
    popup = create_popup()
    COMPUTE_LOOP_ARGS = popup, func
    STILLRUNNING = True
    popup.after(100, compute_loop_popup)
    popup.mainloop()
###


# %%
def get_filenames_labels_and_count(path_data):
    image_subfolders = os.listdir(path_data)
    labels = [label[label.index('-')+1:].replace('_', ' ')
              for label in image_subfolders]

    images_names = {label: os.listdir(path_data + subfolder)
                    for label, subfolder in zip(labels, image_subfolders)}
    n_images_per_class = Series([len(images_names[label])
                                 for label in labels],
                                index=labels)
    n_images_per_class = n_images_per_class.sort_values(ascending=False)
    return {'filenames': images_names,
            'labels': labels,
            'n images per class': n_images_per_class}


def display_n_images_per_class(infos):
    n_images_per_class = DataFrame(infos['n images per class'],
                                   columns=["Nombre d'images"])
    display(n_images_per_class.T)


def plot_classes(infos, n_kept_classes):
    n_images_per_class = infos['n images per class']
    print(f"Nombre de classes: {len(n_images_per_class)}")
    print(f"Nombre total d'images: {n_images_per_class.sum()}")

    print(f"Nombre de classes conservées: {n_kept_classes}")
    print("Nombre total d'images: "
          + f"{n_images_per_class.values[:n_kept_classes].sum()}")
    y = n_images_per_class.values.ravel()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(y, 'o', label='données initiales')
    ax.plot(y[:n_kept_classes], 'ro', label='données conservées')
    ax.legend()
    ax.set_title("Nombre d'images par classe")
    ax.set_xlabel('numéro de classe')
    ax.set_ylabel("nombre d'images")
    ax.annotate(f'{n_kept_classes} classes conservées', [21, 210], color='r',
                va='center', ha='left')

    fig.tight_layout()
    graph_tools.savefig(fig, PATHS_PRINT['explore'] + 'n_images_per_classe')


def __get_label_folder__(label, path_data):
    image_subfolders = os.listdir(path_data)
    for i, folder in enumerate(image_subfolders):
        if label.replace(' ', '_') in folder:
            return folder
    return -1


def get_kept_classes_and_folders(infos, path_data, n_kept_classes):
    n_images_per_class = infos['n images per class']
    labels = n_images_per_class.index.values[:n_kept_classes]
    kept_classes = n_images_per_class.iloc[:n_kept_classes].index.values
    df_label_folder = pd.DataFrame({'folder': str, 'n images': int}, index=[])
    for label in labels:
        folder = __get_label_folder__(label, path_data)
        if folder == -1:
            print_error(f'folder not found for label "{label}"')
            continue
        n_images = n_images_per_class[label]
        df_label_folder.loc[label] = folder, n_images
    infos['df kept classes'] = df_label_folder


def read_image_shapes(path_data, infos):
    shapes = []
    n_bw = 0
    for folder in infos['df kept classes']['folder']:
        path = os.path.join(path_data, folder)
        for filename in os.listdir(path):
            shape = read_image(os.path.join(path, filename)).shape
            # print(os.path.join(path, filename))
            # print(shape)
            shapes.append(shape)
            if len(shape) < 3 or shape[0] < 3:
                n_bw += 1
        infos['image shapes'] = np.asarray(shapes)
    print(f"Nombre d'images en noir et blanc {n_bw}")
    return


def plot_image_shapes(infos):
    shapes = infos['image shapes']
    w = np.unique(shapes[:, 2])
    h = np.unique(shapes[:, 1])
    print(f"Nombre d'images: {shapes.shape[0]}")
    print(f'Nombre de largeurs différentes: {len(w)}')
    print(f'Nombre de hauteurs différentes: {len(h)}')
    arr_count = np.zeros((len(h), len(w)), dtype=int)
    for shape in shapes:
        i = np.where(h == shape[1])[0][0]
        j = np.where(w == shape[2])[0][0]
        arr_count[i, j] += 1
    fig, ax = plt.subplots(figsize=(5, 5))
    # im = ax.imshow(arr_count, cmap='hot')
    im = ax.imshow(np.log10(arr_count + 1e-6), cmap='hot', vmin=np.log10(0.6))
    ax.grid(visible=False)
    # cb = fig.colorbar(im, shrink=0.5, label="nombre d'images")
    cb = fig.colorbar(im, shrink=0.5, label="log10(nombre d'images)")

    i, j = np.unravel_index(arr_count.argmax(), arr_count.shape)
    # print('h, w max:', h[i], w[j])
    ax.annotate(f'{w[j]} x {h[i]} : {arr_count[i, j]} images', xy=[j, i],
                xytext=[20, 80], arrowprops={'color': 'w',
                                             'shrink': 0.05,
                                             'width': 1,
                                             'headwidth': 7,
                                             'headlength': 7})

    ax.set_xlabel('largeur (pixels)')
    indexes = np.linspace(0, w.size-1, 5).astype('int')
    ax.set_xticks(indexes)
    ax.set_xticklabels(w[indexes])

    ax.set_ylabel('hauteur (pixels)')
    indexes = np.linspace(0, h.size-1, 5).astype('int')
    ax.set_yticks(indexes)
    ax.set_yticklabels(h[indexes])

    ax.set_title("décompte du nombre d'image par taille")
    fig.tight_layout()
    graph_tools.savefig(fig, PATHS_PRINT['explore'] + 'count_shapes')
    fig.savefig(PATHS_PRINT['explore'] + 'count_shapes.png')


def __prepare_image__(image: tensor, target_shape: tuple) -> tensor:
    """Resize input tensor corresponding to image according to
       the desired shape, and apply a gaussian blur

    Args:
        image (tensor): input image
        target_shape (tuple): desired shape

    Returns:
        tensor: resized_image
    """
    image = transforms.functional.autocontrast(image)
    image = transforms.functional.equalize(image)
    ht, wt = target_shape
    d, h, w = image.shape

    scale = max(image.shape) / max(target_shape)
    kernel_size = int(5*scale+0.5)
    gaussianBlur = transforms.GaussianBlur(kernel_size + (kernel_size+1) % 2,
                                           scale)
    image = gaussianBlur(image)

    image_f = image.float()
    avg = image_f.mean((1, 2))
    std = image_f.std((1, 2))

    if h > w:
        resize = transforms.Resize((ht, int(ht*w/h+0.5)), antialias=True)
        resized = resize(image)
        crop = transforms.CenterCrop((ht, wt))
        cropped = crop(resized)
    else:
        resize = transforms.Resize((int(wt*h/w+0.5), wt), antialias=True)
        resized = resize(image)
        crop = transforms.CenterCrop((ht, wt))
        cropped = crop(resized)
    # gaussianBlur = transforms.GaussianBlur(5, 1.)
    return cropped, avg, std


def load_resize_save_images(infos: dict, path_load: str, path_save: str,
                            target_shape: tuple):
    n = 0
    n_tot = infos['df kept classes']['n images'].sum()
    df_kept_classes = infos['df kept classes']
    # resize = transforms.Resize(target_shape, antialias=False)
    avgs_std = torch.empty((n_tot, 2, 3))
    for folder, label in zip(df_kept_classes['folder'],
                             df_kept_classes.index.values):
        path = os.path.join(path_load, folder)
        path_save_tmp = path_save + label.replace(' ', '_')
        if not os.path.isdir(path_save_tmp):
            os.makedirs(path_save_tmp)
        for filename in os.listdir(path):
            if (n % 10) == 0:
                print(f"{n/n_tot:.2%}", end='\r')
            image = read_image(os.path.join(path, filename)).to(DEVICE)
            # image = resize(image)
            image, avg, std = __prepare_image__(image, target_shape)
            avgs_std[n, 0, :] = avg[:]
            avgs_std[n, 1, :] = std[:]
            # print('image:', image.dtype)
            torch.save(image.to('cpu'),
                       os.path.join(path_save_tmp, filename[:-3]) + 'pt')
            n += 1
        torch.save(avgs_std, os.path.join(path_save, 'avgs_std.pt'))
    print('100%   ')


def create_annotations_files(infos: dict, path_data: str,
                             train_ratio: float = 0.8):
    kept_classes = infos['df kept classes']
    # display(kept_classes)
    df_annot_file_train = pd.DataFrame({'filename': str(), 'label': int(),
                                        'i': int()},
                                       index=[])
    df_annot_file_test = df_annot_file_train.copy()
    label_ind_to_str = []

    n = 0
    for i, label in enumerate(kept_classes.index.values):
        label_ind_to_str.append(label)
        folder = label.replace(' ', '_')
        # création d'un DF temporaire pour stocker images et label (int)
        path = os.path.join(path_data, folder)
        filenames = os.listdir(path)
        n2 = n + len(filenames)
        tmp = pd.DataFrame({'filename': [folder + '/' + filename for filename
                                         in filenames],
                            'label': [i,]*len(filenames),
                            'i': np.arange(n, n2)
                            })
        n = n2
        tmp_train, tmp_test = model_selection.train_test_split(
                                                        tmp,
                                                        train_size=train_ratio)
        # display(df_annot_file.iloc[-5:,:])
        df_annot_file_train = pd.concat((df_annot_file_train,
                                         tmp_train)).reset_index(drop=True)
        df_annot_file_test = pd.concat((df_annot_file_test,
                                        tmp_test)).reset_index(drop=True)
    print('train dataset:', df_annot_file_train.shape)
    print('test dataset:', df_annot_file_test.shape)
    print("Nombre total d'images:",
          len(df_annot_file_train) + len(df_annot_file_test))  # vérification
    display(df_annot_file_train.sample(10).style.set_caption('train files'))
    display(df_annot_file_test.sample(10).style.set_caption('test files'))
    df_annot_file_train.to_pickle(ANNOTATION_FILENAME + '_train.pickle')
    df_annot_file_test.to_pickle(ANNOTATION_FILENAME + '_test.pickle')
    Series(label_ind_to_str).to_pickle(LABEL_IND_TO_STR_FILENAME)
    return


def display_annotation_files():
    df_annot_file_train = pd.read_pickle(ANNOTATION_FILENAME + '_train.pickle')
    df_annot_file_test = pd.read_pickle(ANNOTATION_FILENAME + '_test.pickle')
    print('train dataset:', df_annot_file_train.shape)
    print('test dataset:', df_annot_file_test.shape)
    print("Nombre total d'images:",
          len(df_annot_file_train) + len(df_annot_file_test))  # vérification
    display(df_annot_file_train.sample(10).style.set_caption('train files'))
    display(df_annot_file_test.sample(10).style.set_caption('test files'))


def one_hot_encoding_label(y, n_kept_classes):
    return torch.zeros(n_kept_classes,
                       dtype=torch.float).scatter_(0,
                                                   torch.tensor(y),
                                                   value=1)


# À partir de la documentation de pytorch
class CustomImageDataset(Dataset):
    """Class to get images and labels
    An autoencoder mode is also present, for which __getitem__ return the image
    and a copy of it.
    """
    def __init__(self, train_test, img_dir, transform=None,
                 target_transform=None, load_all=False):
        self.img_labels = pd.read_pickle(ANNOTATION_FILENAME + '_'
                                         + train_test + ".pickle")
        self.label_ind_to_str = pd.read_pickle(LABEL_IND_TO_STR_FILENAME)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.n_classes = self.img_labels.iloc[:, 1].max() + 1
        self.autoencoder_moder = False
        print('Nombre de classes dans le dataset:', self.n_classes)
        self.__getitem_current__ = self.__getitem_0__

        self.load_all = load_all
        # memory_allocate = torch.cuda.memory_allocated() * (1e-6 / 8)
        # print('GPU memory (MB):', int(memory_allocate))
        # avgs_stds = torch.load(os.path.join(img_dir, 'avgs_std.pt'),
        #                        map_location=DEVICE)
        # print('avgs_std:', avgs_stds.shape)
        release_gpu_cache()
        if load_all:
            self.Xs = []
            for idx in range(len(self)):
                img_path = os.path.join(self.img_dir,
                                        self.img_labels.iloc[idx, 0])
                image = torch.load(img_path, map_location='cpu').float()
                image /= image.max()
                # avg, std = avgs_stds[self.img_labels.iloc[idx, 2]]
                # image = (image-avg.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
                # print('image std:', image.std())
                self.Xs.append(image)
        # memory_allocate = torch.cuda.memory_allocated() * (1e-6 / 8)
        # print('GPU memory (MB):', int(memory_allocate))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.__getitem_current__(idx)

    def __getitem_0__(self, idx):
        if self.load_all:
            image = self.Xs[idx].detach().clone().to(DEVICE)
        else:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = torch.load(img_path, map_location=DEVICE).float()
            image /= image.max()
        i_label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        label = one_hot_encoding_label(i_label, self.n_classes).to(DEVICE)
        return image, label

    def __getitem_autoencodeur__(self, idx):
        # TODO : UPDATE
        if self.load_all:
            image = self.Xs[idx].detach().clone()  # .to(DEVICE)
        else:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = torch.load(img_path, map_location=DEVICE).float()
            image /= image.max()
        if self.transform:
            image = self.transform(image)
        return image, image.clone()

    def set_autoencoder_mode(self, mode: bool):
        self.autoencoder_moder = mode
        if mode:
            self.__getitem_current__ = self.__getitem_autoencodeur__
            return
        self.__getitem_current__ = self.__getitem_0__

    def get_label_str(self, i_label):
        return self.label_ind_to_str[i_label]

    def normalize_images(self):
        for idx in range(len(self)):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = torch.load(img_path, map_location=DEVICE)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            torch.save(normalize(image.float()), img_path)


def test_class_dataset(dataset):
    image, label = dataset[10]
    # print('image type:', type(image), image.max())
    # print('label:', label, 'argmax:', label.argmax().item())
    label = dataset.get_label_str(label.argmax().item())
    image = image.swapaxes(0, 2).swapaxes(0, 1).to('cpu')
    fig, ax = plt.subplots(figsize=(5, 5*image.shape[0]/image.shape[1]))
    ax.imshow(image / image.max())
    ax.set_title(label)
    ax.grid(visible=False)
    ax.axis('off')
###


def define_data_loaders(train_data, test_data):
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                                 shuffle=True)
    return {'train': train_dataloader, 'test': test_dataloader}


def reload_trainer(trainer):
    dataloaders = trainer.dataloaders
    model = trainer.model
    mode = trainer.mode
    filename = 'tmp_reload'
    trainer.save_model(filename)
    del trainer
    trainer = NN_Trainer(model, dataloaders, mode)
    trainer.load_model(filename)
    os.remove(os.path.join(PATH_MODELS + filename.replace(' ', '_')))
    return trainer


def load_trainer_results(filename: str):
    filename = os.path.join(PATH_RESULTS, filename)
    return load_pickle(filename)


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
            print_error('set optimizer before scheduler')

        scheduler = get_scheduler(self.optimizer, scheduler_info)
        if scheduler is None:
            print_error('Scheduler not set')
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
            print_error('Optimizer not set')
            return
        self.optimizer_info = optimizer_info
        self.optimizer, self.need_closure = optimizer

    def save_results(self, filename: str):
        filename = os.path.join(PATH_RESULTS, filename.replace(' ', '_'))
        if self.mode == 'classifier':
            metric_label = 'Accuracy'
        else:
            metric_label = 'MSE'
        to_pickle(filename, {'metric label': metric_label,
                             'metric': self.metrics,
                             'loss': self.losses,
                             'lr': self.lrs,
                             'confusion': self.confusions,
                             'compute time': self.compute_time})

    def plot_lr_scheduler(self, title, savename, epochs):
        # model_state = copy.deepcopy(self.model.state_dict())
        # optimizer = self.optimizer
        # optimizer_state = copy.deepcopy(optimizer.state_dict())
        # scheduler = self.scheduler
        # scheduler_state = copy.deepcopy(scheduler.state_dict())
        # last_lr = optimizer.param_groups[0]['lr']
        # last_epoch = scheduler.last_epoch
        # print('LR begin:', optimizer.param_groups[0]['lr'])
        # print('last epoch:', scheduler.last_epoch)

        model = nn.Linear(in_features=1, out_features=1)
        optimizer = self.optimizer_info
        name = optimizer['name'].lower()
        optimizer = optimizers[name](model.parameters(),
                                     **optimizer['params'])
        scheduler = self.scheduler_info
        name = scheduler['name'].lower()
        scheduler = schedulers[name](optimizer,
                                     **scheduler['params'])

        lrs = []
        # self.model.eval()
        optimizer.zero_grad()
        for i in range(epochs):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        # reload previous state
        # scheduler.load_state_dict(scheduler_state)
        # scheduler.last_epoch = last_epoch
        # optimizer.load_state_dict(optimizer_state)
        # optimizer.param_groups[0]['lr'] = last_lr
        # self.model.load_state_dict(model_state)
        # # ! TODO: search why is the model updated even with eval and zero_grad

        # print('LR end:', self.optimizer.param_groups[0]['lr'])

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(range(epochs), lrs)
        ax.set_xlabel('epoch')
        ax.set_ylabel('learning rate')
        ax.set_title(title)
        fig.tight_layout()
        graph_tools.savefig(fig, PATHS_PRINT['essais'] + savename)

    def load_results(self, filename: str):
        data = load_trainer_results(filename)
        if data is None:
            return
        self.metrics = data['metric']
        self.losses = data['loss']
        self.lrs = data['lr']
        self.confusions = data['confusion']

    def save_model(self, filename: str):
        filename = os.path.join(PATH_MODELS, filename.replace(' ', '_'))
        to_save = {
            'mode': self.mode,
            # 'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            # 'loss': loss,
            'metric': self.metrics,
            'loss': self.losses,
            'lr': self.lrs,
            'confusion': self.confusions,
            'compute time': self.compute_time
            }
        if self.optimizer is not None:
            to_save['optimizer_info']: self.optimizer_info
            to_save['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            to_save['scheduler_state_dict'] = self.scheduler.state_dict()
            to_save['scheduler_info'] = self.scheduler_info
            to_save['last_epoch'] = self.scheduler.last_epoch
        torch.save(to_save, filename)

    def load_model(self, filename: str):
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
            self.scheduler_info = checkpoint['scheduler_info']
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scheduler.last_epoch = checkpoint['last_epoch']
        else:
            self.scheduler = None
            self.scheduler_info = None
        self.metrics = checkpoint['metric']
        self.losses = checkpoint['loss']
        self.lrs = checkpoint['lr']
        self.confusions = checkpoint['confusion']
        self.compute_time = checkpoint['compute time']
        self.set_mode(checkpoint['mode'])

    def train_network(self, epochs: int, lr_mode: str = 'scheduler',
                      init_lr: float = 1e-3):
        if len(self.lrs) >= epochs:
            print(f'current number of epochs ({len(self.lrs)}) >= {epochs}')
            return
        if self.optimizer is None:
            print_error('optimizer not defined')
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

        # self.__train_network_iter__()
        # return
        # run_with_stop_button(self.__train_network_iter__)
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
        metric_tr_plot = self.metrics['train'][max(0, t-5): t]
        metric_ts_plot = self.metrics['test'][max(0, t-5): t]
        metric_tr_plot = [metric_frmt.format(value)
                          for value in metric_tr_plot]
        metric_ts_plot = [metric_frmt.format(value)
                          for value in metric_ts_plot]

        delattr(self, 'tmp')
        clear_output(wait=True)
        print(f'Epoch {t} / {epochs}, '
              + 'lr {:.5f}'.format(self.optimizer.param_groups[0]['lr']))
        print(f'train {metric_lbl}:', metric_tr_plot)
        print(f'test {metric_lbl}:', metric_ts_plot)
        print(f"\nDone! ({t-n_epoch_0} epochs in {chrono(tend-t0)})")

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

        metric_tr, loss_tr, cm_tr = __test_loop__(self.dataloaders['train'],
                                                  self.model, update_correct,
                                                  loss_fn, confusion)

        metric_ts, loss_ts, cm_ts = __test_loop__(self.dataloaders['test'],
                                                  self.model, update_correct,
                                                  loss_fn, confusion)
        # release_gpu_cache()

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

            (metric_tr, loss_tr,
                cm_tr) = __test_loop__(self.dataloaders['train'],
                                       self.model, update_correct,
                                       loss_fn, confusion)

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
        # release_gpu_cache()

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


def conv2d_layer(n_input: int, n_filters: int, n_sub_layers: int,
                 kernel_size: int = 3, stride: tuple = (1, 1),
                 padding: tuple = (1, 1),
                 activation=nn.ReLU(inplace=True)) -> tuple:
    # print('layer:', layer, 'n_input:', n_input, 'n_filters:', n_filters)
    return (
            nn.Conv2d(n_input, n_filters, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(n_filters),
            activation) \
        + (
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(n_filters),
            activation,
            ) * (n_sub_layers-1) \
        + (nn.MaxPool2d(kernel_size=2, stride=2, padding=0),)


def convtransposed2d_layer(n_input: int, n_filters: int, n_sub_layers: int,
                           kernel_size: int = 3, stride: tuple = (1, 1),
                           padding: tuple = (1, 1),
                           activation=nn.ReLU(inplace=True)) -> tuple:
    # TODO : calculer scale_factor
    # if layer == 0:
    #     n_input = 64
    #     n_filters = 3
    # else:
    #     n_input = 64 * (2**(layer))
    #     n_filters = n_input//2
    # print('layer:', layer, 'n_input:', n_input, 'n_filters:', n_filters)
    return (
            nn.ConvTranspose2d(n_input, n_filters, kernel_size=kernel_size,
                               stride=stride, padding=padding,
                               bias=False),  # redundant with batch norm
            nn.BatchNorm2d(n_filters),
            activation) \
        + (
            nn.ConvTranspose2d(n_filters, n_filters, kernel_size=kernel_size,
                               stride=stride, padding=padding,
                               bias=False),  # redundant with batch norm
            nn.BatchNorm2d(n_filters),
            activation,
            ) * (n_sub_layers-1) \
        + (nn.Upsample(scale_factor=(2, 2), mode='bilinear'),)


def get_conv_resize(size_in, kernel, stride, padding):
    return 1 + int((size_in - kernel + 2*padding)/stride)


def get_RU_resize(size_in, stride):
    size_out = get_conv_resize(size_in, 3, stride, 1)
    size_out = get_conv_resize(size_out, 3, 1, 1)
    return size_out


def init_dense_linear(layer):
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.constant_(layer.bias, 0.)


def Conv2d_BN_Activation(n_input: int, n_filters: int,
                         kernel_size: Union[int, Tuple[int, ...]],
                         stride: int, padding: int, activation: Callable):
    # no bias for conv to avoid redondancy with BatchNorm
    cl = nn.Conv2d(n_input, n_filters, kernel_size=kernel_size, stride=stride,
                   padding=padding, bias=False)
    init_he(cl)
    bn = nn.BatchNorm2d(n_filters)
    return nn.Sequential(cl, bn, activation)


def SepConv2d_space(n_input: int, n_filters: int, kernel_sizes: Tuple[int],
                    strides: Tuple[int], paddings: Tuple[int],
                    activation: Callable):
    groups = min(n_input, n_filters)
    layers = []
    for kernel_size, stride, padding in zip(kernel_sizes, strides,
                                            paddings):
        cli = nn.Conv2d(n_input, n_filters, kernel_size=(kernel_size, 1),
                        stride=(stride, 1), padding=(padding, 0),
                        bias=False, groups=groups)
        init_he(cli)
        clii = nn.Conv2d(n_filters, n_filters, kernel_size=(1, kernel_size),
                         stride=(1, stride), padding=(0, padding),
                         bias=False, groups=groups)
        init_he(clii)
        layers += [cli, clii]
        n_input = n_filters
    layers += [nn.BatchNorm2d(n_filters), activation]
    return nn.Sequential(*layers)


activations = {
    'ReLU': nn.ReLU(inplace=True),
    'LReLU': nn.LeakyReLU(inplace=True),
    'Sigmoid': nn.Sigmoid(),
    'SiLU': nn.SiLU(inplace=True),
}


def AlexNet_based_architecture(n_out: int, activation: str,
                               separable_conv_begin: bool = False,
                               separable_conv2_begin: bool = False,
                               ResidualUnits: bool = False,
                               RU_params: dict = {'n_subspace': 128, 'C': 32},
                               SEblock: bool = False,
                               endPoolType: str = 'max'):
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
        conv0 = SepConv2d_space(3, n_filters_in, kernel_sizes=(7, 5),
                                strides=(4, 1), paddings=(0, 0),
                                activation=activation)
    else:
        conv0 = Conv2d_BN_Activation(3, n_filters_in, kernel_size=11, stride=4,
                                     padding=0, activation=activation)
    # shape: 54 x 54
    maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    lrn0 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.)

    # shape: 26 x 26
    if separable_conv2_begin:
        conv2 = SepConv2d_space(n_filters_in, n_filters_2,
                                kernel_sizes=(3, 3, 3),
                                strides=(1, 1, 1), paddings=(0, 0, 1),
                                activation=activation)
    else:
        conv2 = Conv2d_BN_Activation(n_filters_in, n_filters_2, kernel_size=5,
                                     stride=1, padding=0,
                                     activation=activation)
    # shape: 22 x 22
    maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    lrn3 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.)

    # shape: 10 x 10
    if ResidualUnits:
        conv4 = ResidualUnit(n_filters_2, n_filters_3, activation=activation,
                             add_SEblock=SEblock,
                             **RU_params)
    else:
        conv4 = Conv2d_BN_Activation(n_filters_2, n_filters_3, kernel_size=3,
                                     stride=1, padding=1,
                                     activation=activation)

    # shape: 10 x 10
    if endPoolType.lower() == 'max':
        # endPool5 = nn.MaxPool2d(kernel_size=10, stride=1, padding=0)
        endPool5 = nn.AdaptiveMaxPool2d(output_size=1)
    elif endPoolType.lower() == 'average':
        # endPool5 = nn.AvgPool2d(kernel_size=10, stride=1, padding=0)
        endPool5 = nn.AdaptiveAvgPool2d(output_size=1)

    # shape: 1 x 1
    dense6 = nn.Linear(in_features=n_filters_3, out_features=128, bias=False)
    init_he(dense6)
    bn6 = nn.BatchNorm1d(128)
    dense7 = nn.Linear(in_features=128, out_features=128, bias=False)
    init_he(dense7)
    bn7 = nn.BatchNorm1d(128)
    dense8 = nn.Linear(in_features=128, out_features=n_out)
    init_dense_linear(dense8)

    return nn.Sequential(
        conv0,
        maxPool1,
        # lrn0,
        conv2,
        maxPool3,
        # lrn3,
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


def AlexNet_based_architecture_reduced(n_out, greyscale: bool):
    n_input = 1 if greyscale else 3

    def init_he(layer):
        with torch.no_grad():
            # He, recommanded with (leaky) relu only (pytorch)
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in',
                                     nonlinearity='leaky_relu')
            # nn.init.constant_(layer.bias, 0.)

    # shape: 224 x 224
    cl0 = nn.Conv2d(n_input, 96, kernel_size=11, stride=4, padding=0)
    LReLU0 = nn.LeakyReLU(inplace=True)
    init_he(cl0)
    # shape: 54 x 54
    maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 26 x 26
    cl2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
    LReLU2 = nn.LeakyReLU(inplace=True)
    init_he(cl2)
    # shape: 26 x 26
    maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 12 x 12
    cl4 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
    LReLU4 = nn.LeakyReLU(inplace=True)
    init_he(cl4)

    # shape: 12 x 12
    cl5 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
    LReLU5 = nn.LeakyReLU(inplace=True)
    init_he(cl5)

    # shape: 12 x 12
    cl6 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
    LReLU6 = nn.LeakyReLU(inplace=True)
    init_he(cl6)
    maxPool7 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 5 x 5
    dense8 = nn.Linear(in_features=5*5*256, out_features=512)
    LReLU8 = nn.LeakyReLU(inplace=True)
    init_he(dense8)
    dense9 = nn.Linear(in_features=512, out_features=512)
    LReLU9 = nn.LeakyReLU(inplace=True)
    init_he(dense9)
    dense10 = nn.Linear(in_features=512, out_features=n_out)
    # Softmax10 = nn.Softmax
    # init_dense_linear(dense10)

    return nn.Sequential(
        cl0, LReLU0, maxPool1,
        cl2, LReLU2, maxPool3,
        cl4, LReLU4,
        cl5, LReLU5,
        cl6, LReLU6, maxPool7,
        nn.Flatten(),
        dense8, LReLU8,
        nn.Dropout(p=0.5, inplace=False),
        dense9, LReLU9,
        nn.Dropout(p=0.5, inplace=False),
        dense10  # Softmax10
    )


def init_he(layer):
    with torch.no_grad():
        # He, recommanded with (leaky) relu only (pytorch)
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in',
                                 nonlinearity='leaky_relu')
        # nn.init.constant_(layer.bias, 0.)


class SeparateConv2dRGB(nn.Module):
    def __init__(self, n_spatial_filters: int,
                 n_out_channels: int, kernel_size: int, stride: int,
                 padding: int, activation: Callable):
        super().__init__()

        # Define the parameters of the convolutional layer
        self.n_spatial_filters = n_spatial_filters
        self.n_out_channels = n_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cl_s = nn.Conv2d(1, n_spatial_filters, kernel_size,
                              stride=stride, padding=padding)

        # Initialize the weights and bias with random values
        init_he(self.cl_s)
        # nn.init.kaiming_uniform_(self.cl_s.weight, mode='fan_in',
        #                          nonlinearity='leaky_relu')
        # nn.init.zeros_(self.cl_s.bias)

        self.bn = nn.BatchNorm2d(n_spatial_filters)

        self.cl_c = nn.Conv2d(n_spatial_filters, n_spatial_filters,
                              kernel_size=1,
                              stride=1, padding=0)
        nn.init.kaiming_uniform_(self.cl_c.weight, nonlinearity='leaky_relu')
        # nn.init.zeros_(self.cl_c.bias)
        # self.avgPool = nn.AvgPool2d(kernel_size, stride, padding)

        # self.bn_c = nn.BatchNorm2d(n_out_channels)
        self.activation = activation

    def forward(self, x):
        n_x = get_conv_resize(x.shape[-1], self.kernel_size,
                              self.stride, self.padding)
        n_y = get_conv_resize(x.shape[-2], self.kernel_size,
                              self.stride, self.padding)

        n_c = self.n_out_channels
        # out = torch.empty((x.shape[0], n_c+3, n_y, n_x), device=DEVICE)
        # out[:, :n_c, :, :] = self.cl_s(x[:, :1, :, :])
        # out[:, :n_c, :, :] += self.cl_s(x[:, 2:3, :, :])
        # out[:, :n_c, :, :] += self.cl_s(x[:, 1:2, :, :])
        out = self.cl_s(x[:, :1, :, :])
        out += self.cl_s(x[:, 2:3, :, :])
        out += self.cl_s(x[:, 1:2, :, :])
        # out[:, -3:, :, :] = self.avgPool(x)
        out = self.bn(out)
        out = self.cl_c(out)
        out = self.activation(out)
        return out


class ResidualUnit(nn.Module):
    def __init__(self, n_input: int, n_out: int, n_subspace: int,
                 C: int, activation: Callable, add_SEblock: bool):
        super().__init__()
        self.n_input = n_input
        self.n_out = n_out
        self.n_subspace = n_subspace
        self.groups = int(n_subspace // C)

        self.cl0 = nn.Conv2d(n_input, n_subspace, kernel_size=1,
                             stride=1, padding=0, bias=False)
        init_he(self.cl0)
        self.bn0 = nn.BatchNorm2d(n_subspace)

        self.cl1 = nn.Conv2d(n_subspace, n_subspace, kernel_size=3,
                             stride=1, padding=1,
                             bias=False, groups=self.groups)
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
        self.dense0 = nn.Linear(n_channels, int(n_channels//16))
        # self.activation1 = nn.ReLU(inplace=True)
        # self.activation1 = nn.LeakyReLU(inplace=True)
        self.bn0 = nn.BatchNorm1d(int(n_channels//16))
        self.activation1 = activations['SiLU']
        self.dense2 = nn.Linear(int(n_channels//16), n_channels)
        self.activation2 = nn.Sigmoid()

        init_dense_linear(self.dense0)
        init_dense_linear(self.dense2)

    def forward(self, x):
        # print('SEBLOCK')
        # out = self.avgPool(x)
        average = x.mean((-1, -2))
        # print('x:', x.shape, '\naverage:', average.shape)
        out = self.dense0(average)
        # print('out:', out.shape)
        out = self.bn0(out)
        out = self.activation1(out)
        out = self.dense2(out)
        # print('out:', out.shape)
        out = self.activation2(out)
        return out.reshape(out.shape + (1, 1))


class DenseStack(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_layers: int,
                 n_subspace: int = None, activation: Callable = None,
                 dropout: float = 0.5):
        super().__init__()
        if n_layers == 1:
            cl = nn.Linear(in_features, out_features, bias=True)
            init_he(cl)
            return cl
        if n_subspace is None:
            n_subspace = out_features
        if activation is None:
            activaton = activations['SiLU']
        layers = []
        for i in range(n_layers-1):
            cl = nn.Linear(in_features, n_subspace, bias=False)
            init_he(cl)
            bn = nn.BatchNorm1d(n_subspace)
            layers += [cl, bn, activation]
            if dropout != 0.:
                layers.append(nn.Dropout(p=dropout, inplace=False))
            in_features = n_subspace
        layers.append(nn.Linear(n_subspace, out_features, bias=True))
        self.Sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.Sequential(x)


def optim_architecture_0(n_out):
    # INITIAL SPATIAL / COLOR CONVOLUTION
    n_spatial_filters = 96
    # shape: 224 x 224
    cl0_sep = SeparateConv2dRGB(n_spatial_filters=n_spatial_filters,
                                n_out_channels=n_spatial_filters,
                                kernel_size=11, stride=4, padding=0,
                                activation=nn.LeakyReLU(inplace=True))

    # shape: 54 x 54
    maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 26 x 26
    cl2 = nn.Conv2d(n_spatial_filters, 256, kernel_size=5, stride=1,
                    padding=2)
    LReLU2 = nn.LeakyReLU(inplace=True)
    init_he(cl2)
    # shape: 26 x 26
    maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 12 x 12
    cl4 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
    LReLU4 = nn.LeakyReLU(inplace=True)
    init_he(cl4)

    # shape: 12 x 12
    cl5 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
    LReLU5 = nn.LeakyReLU(inplace=True)
    init_he(cl5)

    # shape: 12 x 12
    cl6 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
    LReLU6 = nn.LeakyReLU(inplace=True)
    init_he(cl6)
    maxPool7 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 5 x 5
    dense8 = nn.Linear(in_features=5*5*256, out_features=512)
    LReLU8 = nn.LeakyReLU(inplace=True)
    init_he(dense8)
    dense9 = nn.Linear(in_features=512, out_features=512)
    LReLU9 = nn.LeakyReLU(inplace=True)
    init_he(dense9)
    dense10 = nn.Linear(in_features=512, out_features=n_out)
    # Softmax10 = nn.Softmax
    # init_dense_linear(dense10)

    return nn.Sequential(
        cl0_sep, maxPool1,
        cl2, LReLU2, maxPool3,
        cl4, LReLU4,
        cl5, LReLU5,
        cl6, LReLU6, maxPool7,
        nn.Flatten(),
        dense8, LReLU8,
        nn.Dropout(p=0.5, inplace=False),
        dense9, LReLU9,
        nn.Dropout(p=0.5, inplace=False),
        dense10  # Softmax10
    )


def optim_architecture_1(n_out):
    # activation = nn.LeakyReLU(inplace=True)
    activation = nn.SiLU(inplace=True)
    # INITIAL SPATIAL / COLOR CONVOLUTION
    # shape: 224 x 224
    n_spatial_filters = 96
    cl0_sep = SeparateConv2dRGB(n_spatial_filters=n_spatial_filters,
                                n_out_channels=n_spatial_filters,
                                kernel_size=11, stride=4, padding=0,
                                activation=activation)

    # shape: 54 x 54
    maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 26 x 26
    cl2 = nn.Conv2d(n_spatial_filters, 256, kernel_size=5, stride=1, padding=0)
    activation2 = activation
    init_he(cl2)
    # shape: 22 x 22
    maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 10 x 10
    sepCL4 = separableConv2d(256, 512, 3, 1, 0, activation=activation)
    # cl4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0,
    #                 groups=256)
    # activation4 = nn.LeakyReLU(inplace=True)
    # cl4_c = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
    # activation4_c = nn.LeakyReLU(inplace=True)
    # init_he(cl4)
    # init_he(cl4_c)

    # shape: 8 x 8
    sepCL5 = separableConv2d(512, 512, 3, 1, 0, activation=activation)
    # cl5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0,
    #                 groups=512)
    # activation5 = nn.LeakyReLU(inplace=True)
    # cl5_c = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
    # activation5_c = nn.LeakyReLU(inplace=True)
    # init_he(cl5)
    # init_he(cl5_c)

    # shape: 6 x 6
    sepCL6 = separableConv2d(512, 512, 3, 1, 0, activation=activation)
    # cl6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0,
    #                 groups=512)
    # activation6 = nn.LeakyReLU(inplace=True)
    # cl6_c = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
    # activation6_c = nn.LeakyReLU(inplace=True)
    # init_he(cl6)
    # init_he(cl6_c)

    # shape: 4 x 4
    maxPool7 = nn.MaxPool2d(kernel_size=4, stride=1, padding=0)

    # shape: 1 x 1
    dense8 = nn.Linear(in_features=512, out_features=512)
    LReLU8 = activation
    init_he(dense8)
    dense9 = nn.Linear(in_features=512, out_features=512)
    LReLU9 = activation
    init_he(dense9)
    dense10 = nn.Linear(in_features=512, out_features=n_out)
    # Softmax10 = nn.Softmax
    # init_dense_linear(dense10)

    return nn.Sequential(
        cl0_sep,
        maxPool1,
        cl2, activation2,
        maxPool3,
        sepCL4,
        sepCL5,
        sepCL6,
        maxPool7,
        nn.Flatten(),
        dense8, LReLU8,
        nn.Dropout(p=0.5, inplace=False),
        dense9, LReLU9,
        nn.Dropout(p=0.5, inplace=False),
        dense10  # Softmax10
    )


# Based on ResNet
def optim_architecture_2(n_out):
    activation = nn.SiLU(inplace=True)
    # activation = nn.ReLU(inplace=True)
    # activation = nn.LeakyReLU(inplace=True)
    # INITIAL SPATIAL / COLOR CONVOLUTION
    # shape: 224 x 224

    add_SEblock = True
    sepConv = True

    cl0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0)
    bn0 = nn.BatchNorm2d(64)

    # shape: 109 x 109
    maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    # shape: 54 x 54
    lrn2 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.)

    # shape: 54 x 54
    ru3 = ResidualUnit(64, 128, stride=2, separableConv=sepConv,
                       activation=activation, add_SEblock=add_SEblock)

    # shape: 27 x 27
    ru4 = ResidualUnit(128, 128, stride=1, separableConv=sepConv,
                       activation=activation, add_SEblock=add_SEblock)

    # shape: 27 x 27
    # ru5 = ResidualUnit(128, 256, stride=2, separableConv=sepConv,
    #                    activation=activation, add_SEblock=add_SEblock)

    # shape: 7 x 7
    # ru6 = ResidualUnit(128, 128, stride=1, separableConv=sepConv,
    #                    activation=activation, add_SEblock=add_SEblock)

    # shape: 3 x 3
    # ru7 = ResidualUnit(128, 256, stride=2, separableConv=sepConv,
    #                    activation=activation, SEblock=add_SEblock)

    # shape: 5 x 5
    # ru8 = ResidualUnit(512, 512, stride=1, separableConv=sepConv,
    #                    activation=activation, SEblock=add_SEblock)

    # shape: 5 x 5
    avgPool9 = nn.AvgPool2d(kernel_size=27, stride=1, padding=0)

    # shape: 1 x 1
    dense10 = nn.Linear(in_features=128, out_features=256)
    init_he(dense10)
    bn10 = nn.BatchNorm1d(256)
    dense11 = nn.Linear(in_features=256, out_features=n_out)
    init_he(dense11)

    return nn.Sequential(
        cl0, bn0, activation,
        maxPool1,
        lrn2,
        ru3, ru4,  # ru5, ru6,  # ru7,  # ru8,
        avgPool9,
        nn.Flatten(),
        nn.Dropout(p=0.3, inplace=False),
        dense10, bn10, activation,
        nn.Dropout(p=0.4, inplace=False),
        dense11
    )


def optim_architectures(i: int, n_kept_classes: int):
    architecture = [optim_architecture_0, optim_architecture_1,
                    optim_architecture_2][i]
    return architecture(n_kept_classes)


class NeuralNetwork(nn.Module):
    def __init__(self, sequential=None, filename: str = None):
        super().__init__()
        self.breakpoint = None
        if filename is None:
            self.sequential = sequential
        else:
            self.load_from_file(filename)
        release_gpu_cache()
        self.to(DEVICE)

    def forward(self, x):
        # print('x:', x.shape)
        if x.device != DEVICE:
            x = x.to(DEVICE)
        return self.sequential(x)

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
                        transforms.RandomCrop(size=target_shape),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomRotation(degrees=30),
                        transforms.RandomAffine(degrees=30,
                                                translate=(0.2,)*2,
                                                scale=(0.9, 1.1),
                                                shear=None,
                                                interpolation=BILINEAR),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                        ])
    return transform


optimizers = {'sgd': torch.optim.SGD,
              'adadelta': torch.optim.Adadelta,
              'adagrad': torch.optim.Adagrad,
              'adam': torch.optim.Adam,
              'asgd': torch.optim.ASGD,
              'lbfgs': torch.optim.LBFGS,
              'rmsprop': torch.optim.RMSprop,
              'rprop': torch.optim.Rprop,
              }

schedulers = {
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
    if name not in optimizers:
        print_error(f'optimizer {name} not defined yet')
        return None
    need_closure = str(optimizers[name]).split(' ')[0] in ['LBFGS']
    return (optimizers[name](model.parameters(),
                             **list(optimizer_info.values())[0]),
            need_closure)


def get_scheduler(optimizer, scheduler_info):
    key = list(scheduler_info.keys())[0]
    name = key.lower()
    if name not in schedulers:
        print_error(f'Scheduler {name} not defined yet')
        return
    return schedulers[name](optimizer, **scheduler_info[key])


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

    # mode = mode.lower()
    # if mode == 'classifier':
    #     label = 'Accuracy'

    #     def update_correct(y, pred):
    #         return (pred.argmax(1) == y.argmax(1)).sum().item()

    # else:
    #     label = 'MSE'

    #     def update_correct(y, pred):
    #         return ((pred-X)**2).sum()

    # Evaluating the model with torch.no_grad() ensures that no gradients are
    # computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage
    # for tensors with requires_grad=True
    if confusion:
        true_labels = np.empty(size, dtype=int)
        pred_labels = np.empty_like(true_labels)
        n = 0
    with torch.no_grad():
        # release_gpu_cache()
        for X, y in dataloader:
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


def test_loop_models(dataloader: DataLoader, models: dict,
                     loss_fn=None) -> DataFrame:
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    for model in models.values():
        model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = [0.,] * len(models)
    correct = [0,] * len(models)
    with torch.no_grad():
        for X, y in dataloader:
            for i, model in enumerate(models.values()):
                pred = model(X)
                test_loss[i] += loss_fn(pred, y).item()
                correct[i] += (pred.argmax(1) == y.argmax(1)).sum().item()
    test_loss = np.array(test_loss) / num_batches
    correct = np.array(correct, dtype='float') * (100. / size)
    return DataFrame({'Accuracy (%)': correct, 'loss': test_loss},
                     index=models.keys()).round(2)


def release_gpu_cache():
    torch.cuda.empty_cache()


def print_model_structure(model, name: str):
    print(f'Model "{name}" structure: {model}\n\n')

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()}") # | Values : {param[:2]} \n")


def load_vgg16():
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    print_model_structure(model, 'VGG16')
    return model


# %% TOOLS AND PLOT FUNCS
def plot_examples_for_each_class(path_data, df, n_per_class):
    path_print = os.path.join(PATHS_PRINT['explore'], 'class_examples')
    greyscale = transforms.Grayscale()
    for label in df.index.values:
        folder, N = df.loc[label, :]
        path = os.path.join(path_data, folder)
        filenames = os.listdir(path)
        idxs = np.random.choice(range(N), size=n_per_class, replace=False)
        fig, axs = plt.subplots(nrows=2, ncols=n_per_class, figsize=(8, 3))
        fig.suptitle(label)
        for i, idx in enumerate(idxs):
            image = read_image(os.path.join(path, filenames[idx]))
            grey = greyscale(image)
            axs[0, i].imshow(image.swapaxes(0, 1).swapaxes(2, 1))
            axs[0, i].axis('off')
            axs[1, i].imshow(grey.squeeze(), cmap='gray')
            axs[1, i].axis('off')
        fig.tight_layout()
        graph_tools.savefig(fig, os.path.join(path_print, label))
        fig.savefig(os.path.join(path_print, label + '.png'))


def print_confusion_matrices(results, classes, foldersave):
    for label, results_i in results.items():
        cms = results_i['confusion']

        for i in range(len(results_i['lr'])):
            fig, ax = create_confusion_plot(cms['train'][i], classes, False,
                                            f'train, epoch {i+1}')
            graph_tools.savefig(fig, os.path.join(foldersave,
                                                  f'{label}_train_{i}'))
            plt.close(fig)
            fig, ax = create_confusion_plot(cms['test'][i], classes, False,
                                            f'test, epoch {i+1}')
            graph_tools.savefig(fig, os.path.join(foldersave,
                                                  f'{label}_test_{i}'))
            plt.close(fig)


def plot_accuracies_vs_epochs(title: str, savename: str, results: dict,
                              logy_loss: bool = False,
                              scheduler_info: dict = None,
                              optimizer_info: dict = None):
    fig, axs = plt.subplots(nrows=3, figsize=(5, 8), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1, 1]})
    axs[-1].set_xlabel('epoch')
    axs[1].set_ylabel('pertes')
    # axs_twnx = [axs[0].twinx(), axs[1].twinx()]
    xmax = 0
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax_lr = axs[2]
    ax_lr.set_ylabel('learning rate')
    for i, results_i in enumerate(results.values()):
        metrics = results_i['metric']
        label = results_i['label']
        metric_train = np.asarray(metrics['train'])
        metric_test = np.asarray(metrics['test'])
        if results_i['metric label'] == 'Accuracy':
            metric_train *= 100.
            metric_test *= 100.
        x = np.arange(len(metric_train))

        ax_lr.plot(x[1:], results_i['lr'][1:], color=colors[i])
        axs[0].plot(x, metric_test, color=colors[i], label=label)
        axs[0].plot(x, metric_train, '--', color=colors[i])
        axs[1].plot(x[1:], results_i['loss']['test'][1:], color=colors[i], label=label)
        axs[1].plot(x[1:], results_i['loss']['train'][1:], '--', color=colors[i])
        xmax = max(xmax, x[-1])

    ax_lr.set_ylim([0, ax_lr.get_ylim()[-1]])

    if scheduler_info is not None:
        lrs = get_scheduler_curve(optimizer_info, scheduler_info, x[-1])
        ax_lr.plot(lrs, '--r', label='scheduler')
        ax_lr.legend()

    label = results_i['metric label']
    if label == 'Accuracy':
        axs[0].set_ylabel('prédictions correctes (%)')
        axs[0].set_ylim([0, 100])
    else:
        axs[0].set_ylabel(label)
    for ax in axs:
        ax.set_xticks([int(x) for x in ax.get_xticks()
                       if (x >= 0) and (x < xmax*1.05)])
    fig.suptitle(title)
    if len(results) > 1:
        lgd = axs[0].legend()  # loc='upper left')

    # Create custom legend handles with corresponding line styles
    ln1, = axs[0].plot([], [], linestyle='-', color='w')
    ln2, = axs[0].plot([], [], linestyle='--', color='w')
    axs[1].legend(handles=[ln1, ln2], labels=['test set', 'train set'])

    if logy_loss:
        axs[1].set_yscale('log')
    if len(results) > 1:
        axs[0].add_artist(lgd)
    fig.tight_layout()
    graph_tools.savefig(fig, PATHS_PRINT['essais'] + savename)


def plot_ae_results(model, dataloader, mse, loss):
    ## TODO : MERGE GRAPHS
    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.set_xlabel('epoch')
    # ax.set_ylabel('MSE')
    # ax.plot(np.arange(1, len(mse)+1), mse)
    # ax.set_xticks([int(x+0.5) for x in ax.get_xticks()])
    # ax.set_title("Évolution du MSE durant l'entrainement")
    # fig.tight_layout()

    # fig, ax = plt.subplots(figsize=(5, 3))
    # ax.set_xlabel('epoch')
    # ax.set_ylabel('pertes')
    # ax.plot(np.arange(1, len(loss)+1), loss)
    # ax.set_xticks([int(x+0.5) for x in ax.get_xticks()])
    # ax.set_title("Évolution de la fonction de pertes durant l'entrainement")
    # fig.tight_layout()

    # Set the model to evaluation mode
    model.eval()

    data_iter = iter(dataloader)
    batch = next(data_iter)
    X, labels = batch
    test_data = X[:4].clone()
    with torch.no_grad():
        reconstructed_data = model(test_data)

    test_data = test_data.swapaxes(1, 2).swapaxes(2, 3)
    reconstructed_data = reconstructed_data.swapaxes(1, 2).swapaxes(2, 3)

    for i in range(test_data.shape[0]):
        test_data[i] -= test_data[i].min()
        test_data[i] /= test_data[i].max() + 0.5
        reconstructed_data[i] -= reconstructed_data[i].min()
        reconstructed_data[i] /= reconstructed_data[i].max() + 0.5
    print('test:', test_data.min(), test_data.max())
    print('reconstructed:', reconstructed_data.min(), reconstructed_data.max())
    test_data = test_data.to('cpu').numpy()
    reconstructed_data = reconstructed_data.to('cpu').numpy()

    num_samples = reconstructed_data.shape[0]
    fig, axes = plt.subplots(nrows=num_samples, ncols=2,
                             figsize=(6, 2*num_samples))

    axes[0, 0].set_title('Original')
    axes[0, 1].set_title('Reconstruite')
    for i in range(num_samples):
        axes[i, 0].imshow(test_data[i].squeeze(), cmap='gray')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed_data[i].squeeze(), cmap='gray')
        axes[i, 1].axis('off')

    fig.tight_layout()


def plot_schedulers():
    lr_scheduler = torch.optim.lr_scheduler

    def lmbda(epoch):
        return 0.9 ** epoch

    schedulers_tmp = {
        'multiplicative': (lr_scheduler.MultiplicativeLR,
                           dict(lr_lambda=lmbda)),
        'lambda': (lr_scheduler.LambdaLR,
                   dict(lr_lambda=lmbda)),
        'step': (lr_scheduler.StepLR,
                 dict(step_size=10, gamma=0.5)),
        'multi-step': (lr_scheduler.MultiStepLR,
                       dict(milestones=[30, 50, 70], gamma=0.4)),
        'exponential': (lr_scheduler.ExponentialLR,
                        dict(gamma=0.95)),
        'cosine annealing': (lr_scheduler.CosineAnnealingLR,
                             dict(T_max=30, eta_min=0)),
        'cyclic triangular': (lr_scheduler.CyclicLR,
                              dict(base_lr=0.001, max_lr=100, step_size_up=10,
                                   mode="triangular")),
        'cyclic triangular 2': (lr_scheduler.CyclicLR,
                                dict(base_lr=0.001, max_lr=100,
                                     step_size_up=10, mode="triangular2")),
        'cyclic exp range': (lr_scheduler.CyclicLR,
                             dict(base_lr=0.001, max_lr=100, step_size_up=10,
                                  mode="exp_range", gamma=0.95)),
        'one cycle cos': (lr_scheduler.OneCycleLR,
                          dict(max_lr=100, steps_per_epoch=10, epochs=10)),
        'one cycle linear': (lr_scheduler.OneCycleLR,
                             dict(max_lr=100, steps_per_epoch=10, epochs=10,
                                  anneal_strategy='linear')),
        'cosine annealing warm restart':
            (lr_scheduler.CosineAnnealingWarmRestarts,
             dict(T_0=30, T_mult=1, eta_min=0.001, last_epoch=-1))
    }

    names_groups = [['lambda', 'multiplicative', 'exponential'],
                    ['step', 'multi-step'],
                    ['cosine annealing', 'cosine annealing warm restart'],
                    ['cyclic triangular', 'cyclic triangular 2',
                     'cyclic exp range'],
                    ['one cycle cos', 'one cycle linear']
                    ]

    for i, names in enumerate(names_groups):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_xlabel('epochs')
        ax.set_ylabel('learning rate')

        for name in names:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=100)
            scheduler, params = schedulers_tmp[name]
            scheduler = scheduler(optimizer, **params)
            lrs = []
            for ii in range(100):
                optimizer.step()
                lrs.append(optimizer.param_groups[0]["lr"])
                scheduler.step()
            ax.plot(range(100), lrs, label=name)

        ax.legend()
        fig.tight_layout()
        graph_tools.savefig(fig, PATHS_PRINT['essais'] + f'schedulers_tmp/{i}')


def create_confusion_plot(cm, classes, normalize=False,
                          title='Matrice de confusion', cmap=plt.cm.hot):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='none', cmap=cmap)
    ax.grid(visible=False)
    ax.set_title(title)
    max_cm = cm.max()
    cm_ticks = np.round(np.linspace(0, max_cm, 7))
    fig.colorbar(im, ticks=cm_ticks)

    tick_marks = np.arange(len(classes))
    print('cm:', cm.shape, 'tickmarks:', tick_marks)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         plt.text(j, i, format(cm[i, j], fmt),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('label')
    ax.set_xlabel('prédiction')
    fig.tight_layout()
    return fig, ax


def test_plot_slider():
    filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import ipywidgets as widgets

    # Define the images and corresponding labels
    images = [plt.imread(filename) for filename in filenames]
    labels = ['Label 1', 'Label 2', 'Label 3', ...]  # Replace with your labels

    for image in images:
        print('image:', image.shape)

    # Create the FigureWidget
    fig = go.FigureWidget(make_subplots(rows=1, cols=1))

    # Add the image trace to the figure
    image_trace = go.Image(z=images[0])
    fig.add_trace(image_trace)

    # Define the slider widget
    slider = widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0)

    # Define the update function for the slider
    def update_image(change):
        img_index = change['new']
        with fig.batch_update():
            fig.data[0].z = images[img_index]

    # Register the update function with the slider
    slider.observe(update_image, names='value')

    fig.update_layout(width=800, height=600)  # Adjust width and height as needed
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Display the FigureWidget and slider
    display(widgets.VBox([fig, slider]))


def plot_data_augmentation_example(image, transform):
    image -= image.min()
    image /= image.max()
    path_print = os.path.join(PATHS_PRINT['essais'], 'data_aug')
    fig, ax = plt.subplots(figsize=(3, 0.2+3*image.shape[1]/image.shape[2]))
    ax.imshow(image.to('cpu').swapaxes(0, 1).swapaxes(1, 2))
    ax.axis('off')
    ax.set_title('image originale')
    fig.tight_layout()
    # graph_tools.savefig(fig, os.path.join(path_print, 'original.png'))
    fig.savefig(os.path.join(path_print, 'original.png'), dpi=300)

    fig, axs = plt.subplots(ncols=4, figsize=(9, 3))
    for i, ax in enumerate(axs):
        image_2 = transform(image).to('cpu').swapaxes(0, 1).swapaxes(1, 2)
        image_2 -= image_2.min()
        image_2 /= image_2.max()
        ax.imshow(image_2)
        ax.axis('off')
    fig.suptitle('images transformées')
    fig.tight_layout()
    fig.savefig(os.path.join(path_print, 'transformed.png'), dpi=300)


# %% END OF FILE
###
