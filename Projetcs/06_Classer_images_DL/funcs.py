#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% IMPORT PACKAGES
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time

from sklearn import model_selection

import torch
from torch import nn, tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import utilities_tools_and_graph as graph_tools

PATHS_PRINT = {'explore': 'Figures/explore/'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANNOTATION_FILENAME = 'data/annotation_file'
LABEL_IND_TO_STR_FILENAME = 'data/label_ind_to_str.pickle'
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
            print(f'ERROR: folder not found for label "{label}"')
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


def __image_resize_crop__(image: tensor, target_shape: tuple) -> tensor:
    """Resize input tensor corresponding to image according to
       the desired shape

    Args:
        image (tensor): input image
        target_shape (tuple): desired shape

    Returns:
        tensor: resized_image
    """
    ht, wt = target_shape
    d, h, w = image.shape
    if h > w:
        resize = transforms.Resize((ht, int(ht*w/h+0.5)), antialias=True)
        resized = resize(image)
        crop = transforms.CenterCrop((ht, wt))
        cropped = crop(resized)
    else:
        resize = transforms.Resize((int(wt*h/w+0.5), wt))
        resized = resize(image)
        crop = transforms.CenterCrop((ht, wt))
        cropped = crop(resized)
    return cropped


def load_resize_crop_images(infos: dict, path_load: str, path_save: str,
                            target_shape: tuple):
    n = 0
    n_tot = infos['df kept classes']['n images'].sum()
    df_kept_classes = infos['df kept classes']
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
            # print(image.shape)
            # plt.imshow(image.swapaxes(0,1).swapaxes(2,1))
            image = __image_resize_crop__(image, target_shape)
            # plt.imshow(image.swapaxes(0,1).swapaxes(2,1))
            # print(image.shape)
            torch.save(image,
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
    def __init__(self, train_test, img_dir, transform=None,
                 target_transform=None):
        self.img_labels = pd.read_pickle(ANNOTATION_FILENAME + '_'
                                         + train_test + ".pickle")
        self.label_ind_to_str = pd.read_pickle(LABEL_IND_TO_STR_FILENAME)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.n_classes = self.img_labels.iloc[:, 1].max() + 1
        print('Nombre de classes dans le dataset:', self.n_classes)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = torch.load(img_path, map_location=DEVICE)
        # image = image / image.max()
        i_label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        label = one_hot_encoding_label(i_label, self.n_classes)
        return image, label

    def get_label_str(self, i_label):
        return self.label_ind_to_str[i_label]


def test_class_dataset(dataset):
    image, label = dataset[0]
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


# %% END OF FILE
###
