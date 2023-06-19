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
from typing import Callable
from IPython.display import clear_output, display


from sklearn import model_selection

import torch
from torch import nn, tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import utilities_tools_and_graph as graph_tools

BILINEAR = transforms.InterpolationMode.BILINEAR

PATH_DATA = 'data/'
PATH_RESULTS = PATH_DATA + 'results/'
PATH_MODELS = PATH_DATA + 'models/'
PATHS_PRINT = {'explore': 'Figures/explore/',
               'essais': 'Figures/essais'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANNOTATION_FILENAME = 'data/annotation_file'
LABEL_IND_TO_STR_FILENAME = 'data/label_ind_to_str.pickle'
BATCH_SIZE = 32
# BATCH_SIZE = 8

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
    im = ax.imshow(arr_count, cmap='hot')
    ax.grid(visible=False)
    cb = fig.colorbar(im, shrink=0.5, label="nombre d'images")

    i, j = np.unravel_index(arr_count.argmax(), arr_count.shape)
    print('i,j max:', i, j)
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
    return cropped


def load_resize_save_images(infos: dict, path_load: str, path_save: str,
                            target_shape: tuple):
    n = 0
    n_tot = infos['df kept classes']['n images'].sum()
    df_kept_classes = infos['df kept classes']
    # resize = transforms.Resize(target_shape, antialias=False)
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
            image = __prepare_image__(image, target_shape)
            torch.save(image.to('cpu'),
                       os.path.join(path_save_tmp, filename[:-3]) + 'pt')
            n += 1
    print('100%   ')


def create_annotations_files(infos: dict, path_data: str,
                             train_ratio: float = 0.8):
    kept_classes = infos['df kept classes']
    display(kept_classes)
    df_annot_file_train = pd.DataFrame({'filename': str, 'label': int()},
                                       index=[])
    df_annot_file_test = df_annot_file_train.copy()
    label_ind_to_str = []
    for i, label in enumerate(kept_classes.index.values):
        label_ind_to_str.append(label)
        folder = label.replace(' ', '_')
        # création d'un DF temporaire pour stocker images et label (int)
        path = os.path.join(path_data, folder)
        filenames = os.listdir(path)
        tmp = pd.DataFrame({'filename': [folder + '/' + filename for filename
                                         in filenames],
                            'label': [i,]*len(filenames)})
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
        if load_all:
            self.Xs = []
            for idx in range(len(self)):
                img_path = os.path.join(self.img_dir,
                                        self.img_labels.iloc[idx, 0])
                image = torch.load(img_path).float()  # , map_location=DEVICE
                image /= image.max()
                self.Xs.append(image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.__getitem_current__(idx)

    def __getitem_0__(self, idx):
        if self.load_all:
            image = self.Xs[idx]  # .clone().to(DEVICE)
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
        if self.load_all:
            image = self.Xs[idx].detach().clone().to(DEVICE)
        else:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = torch.load(img_path, map_location=DEVICE).float()
            image /= image.max()
        if self.transform:
            image = self.transform(image)
        return image, image.copy()

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
    print('image type:', type(image), image.max())
    print('label:', label, 'argmax:', label.argmax().item())
    label = dataset.get_label_str(label.argmax().item())
    image = image.swapaxes(0, 2).swapaxes(0, 1).to('cpu')
    fig, ax = plt.subplots(figsize=(5, 5*image.shape[0]/image.shape[1]))
    ax.imshow(image)
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
        self.metrics = []
        self.losses = []
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

    def set_scheduler(self, scheduler: dict):
        if self.optimizer is None:
            print_error('set optimizer before scheduler')
        print('TODO : SET_SCHEDULER')
        ...
        self.scheduler = scheduler(self.optimizer, **scheduler['params'])

    def set_optimizer(self, optimizer: dict):
        """Set the optimizer for the Neural Network training.

        Args:
            optimizer (dict): key 'name' provide the optimizer name
                              must contains 'params' attribute (dict) which
                              corresponds to the parameters of the optimizer
        """
        name = optimizer['name']
        if name.lower() not in optimizers:
            print_error(f'optimizer {name} not defined yet')
            return
        self.optimizer_info = optimizer
        self.optimizer = optimizers[name.lower()](self.model.parameters(),
                                                  **optimizer['params'])
        self.need_closure = str(optimizer).split(' ')[0] in ['LBFGS']

    def save_results(self, filename: str):
        filename = os.path.join(PATH_RESULTS, filename.replace(' ', '_'))
        if self.mode == 'classifier':
            metric_label = 'Accuracy'
        else:
            metric_label = 'MSE'
        to_pickle(filename, {'metric label': metric_label,
                             'metric': self.metrics,
                             'loss': self.losses})

    def load_results(self, filename: str):
        data = load_trainer_results(filename)
        if data is None:
            return
        self.metrics = data['metric']
        self.losses = data['loss']

    def save_model(self, filename: str):
        filename = os.path.join(PATH_MODELS, filename.replace(' ', '_'))
        torch.save({
            'optimizer_info': self.optimizer_info,
            'mode': self.mode,
            # 'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss': loss,
            'metric': self.metrics,
            'loss': self.losses
            }, filename)

    def load_model(self, filename: str):
        filename = os.path.join(PATH_MODELS, filename.replace(' ', '_'))

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.set_optimizer(checkpoint['optimizer_info'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metric']
        self.losses = checkpoint['loss']
        self.set_mode(checkpoint['mode'])

    def train_network(self, epochs: int):
        if self.optimizer is None:
            print_error('optimizer not defined')
            return
        if self.mode == 'classifier':
            loss_fn = nn.CrossEntropyLoss()
            metric_frmt = 'Accuracy {:.2%}'

            def update_correct(y, pred):
                return (pred.argmax(1) == y.argmax(1)).sum().item()
        else:
            loss_fn = nn.MSELoss()
            metric_frmt = 'MSE {:.3f}'

            def update_correct(y, pred):
                return ((pred-X)**2).sum().item()

        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_dataloader = self.dataloaders['train']
        test_dataloader = self.dataloaders['test']

        # Set the model to training mode
        # important for batch normalization and dropout layers
        model.train()

        if len(self.metrics) == 0:
            metric = -1
        else:
            metric = self.metrics[-1]

        size = len(train_dataloader.dataset)
        t0 = time.time()
        for t in range(epochs):
            clear_output(wait=True)
            print(f'Epoch {t+1} / {epochs}, '
                  'current ' + metric_frmt.format(metric))
            __train_loop__(train_dataloader, model, loss_fn, optimizer,
                           self.need_closure)
            release_gpu_cache()  # release cache used by dataloaders
            metric, loss = __test_loop__(test_dataloader, model,
                                         update_correct, loss_fn)
            release_gpu_cache()  # release cache used by dataloaders
            self.metrics.append(metric)
            self.losses.append(loss)
            if scheduler is not None:
                scheduler.step()
        clear_output(wait=True)
        print(f'Epoch {t+1}/{epochs}, '
              ' ' + metric_frmt.format(metric))
        print(f"\nDone! ({epochs} epochs in {chrono(time.time()-t0)})")
        release_gpu_cache()  # release cache used by dataloaders


def conv2d_layer(layer: int, n_sub_layers: int,
                 kernel_size: int = 3, stride: tuple = (1, 1),
                 padding: tuple = (1, 1),
                 activation=nn.ReLU(inplace=True)) -> tuple:
    if layer == 0:
        n_input = 3
        n_filters = 64
    else:
        n_input = 64 * (2**(layer-1))
        n_filters = n_input*2
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


def convtransposed2d_layer(layer: int, n_sub_layers: int,
                           kernel_size: int = 3, stride: tuple = (1, 1),
                           padding: tuple = (1, 1),
                           activation=nn.ReLU(inplace=True)) -> tuple:
    # TODO : calculer scale_factor
    if layer == 0:
        n_input = 64
        n_filters = 3
    else:
        n_input = 64 * (2**(layer))
        n_filters = n_input//2
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


class NeuralNetwork_AE_based(nn.Module):
    def __init__(self, filename=None):
        super().__init__()
        n_sub_layers = 2
        n_layers = 4
        self.n_layers = n_layers
        self.breakpoint = None
        self.flatten = nn.Flatten()
        conv_layers = []
        for layer in range(n_layers):
            conv_layers += conv2d_layer(layer, n_sub_layers)
        self.conv_stack = nn.Sequential(  # input size 224 x 224
            *conv_layers
        )
        convT_layers = []
        for layer in range(n_layers):
            convT_layers = list(convtransposed2d_layer(layer, n_sub_layers))\
                           + convT_layers
        self.convT_stack = nn.Sequential(*convT_layers)
        self.mode_AE = True
        if filename is not None:
            self.load_from_file(filename)
        self.to(DEVICE)

    def drop_decoder(self):
        delattr(self, 'convT_stack')
        release_gpu_cache()

    def lock_encoder(self):
        for param in self.conv_stack.parameters():
            param.requires_grad = False

    def unlock_encoder(self):
        for param in self.conv_stack.parameters():
            param.requires_grad = True

    def to_classifier(self, n_out):
        self.n_out = n_out
        n_wght_per_layer_classifier = 1024
        self.mode_AE = False
        # for module in reversed(self.conv_stack.modules()):
        #     if isinstance(module, nn.Conv2d):
        #         n_filters_out_conv = module.out_channels
        #         break
        n_filters_out_conv = 64 * (2**(self.n_layers-1))
        self.avg_pool = nn.AdaptiveAvgPool2d((5, 5)).to(DEVICE)
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(5*5*n_filters_out_conv),
            nn.Linear(in_features=5*5*n_filters_out_conv,
                      out_features=n_wght_per_layer_classifier,
                      bias=False),  # redundant with batch norm
            nn.BatchNorm1d(n_wght_per_layer_classifier),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=n_wght_per_layer_classifier,
                      out_features=n_wght_per_layer_classifier,
                      bias=False),  # redundant with batch norm
            nn.BatchNorm1d(n_wght_per_layer_classifier),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=n_wght_per_layer_classifier,
                      out_features=n_out,
                      bias=False)
        ).to(DEVICE)

    def forward(self, x):
        # print('input x:', x.shape)
        y = self.conv_stack(x)
        # print('y after conv:', y.shape)
        if self.mode_AE:
            logits = self.convT_stack(y)
        else:
            y = self.avg_pool(y)
            # print('y after avg pool:', y.shape)
            logits = self.classifier(self.flatten(y))
        return logits

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


class NeuralNetwork(nn.Module):
    def __init__(self, n_out, filename=None):
        super().__init__()
        n_sub_layers = 2
        n_layers = 4
        n_wght_per_layer_classifier = 1024
        self.breakpoint = None
        self.n_out = n_out
        self.flatten = nn.Flatten()
        conv_layers = []
        for layer in range(n_layers):
            conv_layers += conv2d_layer(layer, n_sub_layers)
        self.conv_stack = nn.Sequential(  # input size 224 x 224
            *conv_layers
        )
        n_filters_out_conv = 64 * (2**(n_layers-1))
        self.avg_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(5*5*n_filters_out_conv),
            nn.Linear(in_features=5*5*n_filters_out_conv,
                      out_features=n_wght_per_layer_classifier,
                      bias=False),  # redundant with batch norm
            nn.BatchNorm1d(n_wght_per_layer_classifier),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=n_wght_per_layer_classifier,
                      out_features=n_wght_per_layer_classifier,
                      bias=False),  # redundant with batch norm
            nn.BatchNorm1d(n_wght_per_layer_classifier),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=n_wght_per_layer_classifier,
                      out_features=n_out,
                      bias=False)
        )
        if filename is not None:
            self.load_from_file(filename)
        self.to(DEVICE)

    def forward(self, x):
        # print('input x:', x.shape)
        y = self.conv_stack(x)
        # print('y after conv:', y.shape)
        y = self.avg_pool(y)
        # print('y after avg pool:', y.shape)
        logits = self.classifier(self.flatten(y))
        return logits

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


def base_transform(target_shape):
    return transforms.CenterCrop(target_shape)


def data_augmentation_transform(target_shape):
    transform = transforms.Compose([
                        transforms.RandomCrop(size=target_shape),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomRotation(degrees=30),
                        transforms.RandomAffine(degrees=30,
                                                translate=(0.05,)*2,
                                                scale=(0.9, 1.1),
                                                shear=None,
                                                interpolation=BILINEAR),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                               saturation=0.2, hue=0.1),
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


def train_network_ae(model, epochs, dataloaders, optimizer_info: dict):
    loss_fn = nn.MSELoss()

    optimizer = optimizer_info['name'].lower()
    if optimizer not in optimizers:
        print_error(f'optimizer {optimizer} not defined yet')
        return
    optimizer = optimizers[optimizer](model.parameters(),
                                      **optimizer_info['params'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                gamma=0.1)
    l_mse, losses = [], []
    t0 = time.time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        __train_loop_ae__(dataloaders['train'], model, loss_fn, optimizer)
        mse, loss = __test_loop_ae__(dataloaders['test'], model, loss_fn)
        l_mse.append(mse.item())
        losses.append(loss)
        scheduler.step()
    print(f"Done! ({epochs} epochs in {chrono(time.time()-t0)})")
    release_gpu_cache()  # release cache used by dataloaders
    return l_mse, losses


def __train_loop_ae__(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization
    # and dropout layers

    # print(str(optimizer))
    optimizer_name = str(optimizer).split(' ')[0]
    need_closure = optimizer_name in ['LBFGS']
    # print(optimizer_name, need_closure)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        release_gpu_cache()  # release cache used by dataloaders
        # print('X:', X.device, 'y', y.device)
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        loss.backward()
        if need_closure:
            def closure():
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, X)
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# from pytorch tutorials
def __test_loop_ae__(dataloader, model, loss_fn=None):
    # Set the model to evaluation mode
    # important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    mse = 0.

    # Evaluating the model with torch.no_grad() ensures that no gradients are
    # computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage
    # for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            release_gpu_cache()  # release cache used by dataloaders
            pred = model(X)
            test_loss += loss_fn(pred, X).item()
            mse += ((pred-X)**2).sum()

    test_loss /= num_batches
    mse /= size
    print(f"Test Error: \nMSE: {mse:.5f}, Avg loss: "
          f"{test_loss:>8f} \n")
    return mse, test_loss


def train_network(model, epochs, dataloaders, optimizer_info: dict):
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    optimizer = optimizer_info['name'].lower()
    if optimizer not in optimizers:
        print_error(f'optimizer {optimizer} not defined yet')
        return
    optimizer = optimizers[optimizer](model.parameters(),
                                      **optimizer_info['params'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                gamma=0.1)
    # previous_weights = copy.deepcopy(model.state_dict())
    # previous_accuracy = None
    accuracies, losses = [], []
    t0 = time.time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        __train_loop__(dataloaders['train'], model, loss_fn, optimizer)
        accuracy, loss = __test_loop__(dataloaders['test'], model, loss_fn)
        accuracies.append(accuracy)
        losses.append(loss)
        scheduler.step()
        # if previous_accuracy is None:
        #     previous_accuracy = accuracy
        #     continue
        # if accuracy < previous_accuracy:
        #     # reduce learning reate
        #     new_learning_rate = optimizer.param_groups[0]['lr'] * 0.1
        #     optimizer.param_groups[0]['lr'] = new_learning_rate
        #     # Cancel the current epoch and restore the previous weights
        #     model.load_state_dict(previous_weights)
        # else:
        #     # Save current weights
        #     previous_weights = copy.deepcopy(model.state_dict())
        #     # Update the previous accuracy
        #     previous_accuracy = accuracy
    print(f"Done! ({epochs} epochs in {chrono(time.time()-t0)})")
    release_gpu_cache()  # release cache used by dataloaders
    return accuracies, losses


# updated from pytorch tutorials
def __train_loop__(dataloader, model, loss_fn, optimizer, need_closure):
    size = len(dataloader.dataset)
    # Set the model to training mode
    #  important for batch normalization and dropout layers

    n = 0
    model.train()
    release_gpu_cache()  # release cache used by dataloaders
    for batch, (X, y) in enumerate(dataloader):
        # print('X:', X.device, 'y', y.device)
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X)
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

        n += len(X)
        if batch % 4 == 0:
            loss = loss.item()
            message = f"loss: {loss:>4f}  [{n:>5d}/{size:>5d}]"
            print(message, end='\r')
    if not isinstance(loss, float):
        loss = loss.item()
    message = f"loss: {loss:>4f}  [{size:>5d}/{size:>5d}]"
    print(message)
    del X, y
    release_gpu_cache()  # release cache used by dataloaders


# updated from pytorch tutorials
def __test_loop__(dataloader: DataLoader, model: nn.Module,
                  update_correct: Callable, loss_fn: Callable):
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
    with torch.no_grad():
        release_gpu_cache()  # release cache used by dataloaders
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y.argmax(1)).sum().item()
            correct += update_correct(y, pred)

    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n {label}: {(100*correct):>0.1f}%, Avg loss: "
    #       f"{test_loss:>8f} \n")
    del X, y
    release_gpu_cache()  # release cache used by dataloaders
    return correct, test_loss


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


def plot_accuracies_vs_epochs(title: str, savename: str, results: dict):
    fig, axs = plt.subplots(nrows=2, figsize=(5, 7), sharex=True)
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('pertes')
    xmax = 0
    for results_i in results.values():
        metric = np.asarray(results_i['metric'])*100
        label = results_i['label']
        x = np.arange(1, len(metric)+1)
        axs[0].plot(x, metric, label=label)
        axs[1].plot(x, results_i['loss'], label=label)
        xmax = max(xmax, x[-1])

    label = results_i['metric label']
    if label == 'Accuracy':
        axs[0].set_ylabel('prédictions correctes (%)')
    else:
        axs[0].set_ylabel(label)
    for ax in axs:
        ax.set_xticks([int(x) for x in ax.get_xticks()
                       if (x >= 0) and (x < xmax*1.05)])
    fig.suptitle(title)
    axs[0].legend()
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

# %% END OF FILE
###
