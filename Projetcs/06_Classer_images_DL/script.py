#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
from typing import Callable, Union, Tuple, List, Dict
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

import torch
from torch import nn, tensor
from torch.nn import Dropout
from torchvision.models import densenet161, DenseNet161_Weights
from torchvision.io import read_image
import torchvision.transforms as transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###


# %%

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
        # if len(layer) > 1:
        #     init_he(cl)
        # else:
        #     init_dense_linear(cl)

    def forward(self, x):
        return self.seq(x)

    def weight_scaling(self):
        weight_scaling1d(self.seq[0].weight)

    def __repr__(self):
        # Customize the representation when printed
        representation = "Dense:\n"
        for i, layer in enumerate(self.seq):
            representation += f"  ({i}): {layer.__repr__()}\n"
        return representation


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
        # release_gpu_cache()
        self.to(DEVICE)
        self.seq.to(DEVICE)
        self.fc.to(DEVICE)

    def forward(self, x):
        # print('x:', x.shape)
        if x.device != DEVICE:
            x = x.to(DEVICE)
        return self.fc(self.seq(x))


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
    if len(layers) == 1:
        return layers[0]
    return nn.Sequential(*layers)


# def release_gpu_cache():
#     torch.cuda.empty_cache()


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
                                     std=[0.229, 0.224, 0.225]),
    ])


def predict(img):
    target_shape = (224, 224)
    img_prepared = __prepare_image__(img, target_shape).float()
    img_prepared /= img_prepared.max()
    img_prepared = base_transform(target_shape)(img_prepared)
    img_prepared = img_prepared.reshape((1,) + img_prepared.shape)
    pred = model(img_prepared)
    argsort = pred.argsort().to('cpu')[0].numpy()[::-1]
    pred = pred.argmax(1).item()
    return pred, argsort

###


# %%

activations = {
    'Mish': nn.Mish(inplace=True),
}

model_data = torch.load('data/models/fc_densenet.pth')
classes = model_data['classes']

fc = model_data['fc dict']
# print('fc:\n', fc)
DenseNet161_layers = [
    (
        'densenet161',
        {'weights': DenseNet161_Weights.IMAGENET1K_V1}
    )
]

model = NeuralNetwork(DenseNet161_layers, fc)
model.seq.classifier = nn.Identity()
model.fc.load_state_dict(model_data['fc state dict'])
model.eval()
###

# %%
if False:  # vérification sur jeu d'entrée
    print('classes:', classes)
    n_ok = 0
    n_tot = 0
    path_images = 'data/source/images/'
    for folder in os.listdir(path_images):
        tmp = folder.replace('_', ' ').replace('-', ' ')
        b_in = False
        for i_cl, ci in enumerate(classes):
            if ci in tmp:
                b_in = True
                break
        if not b_in:
            continue
        print('folder:', folder, classes[i_cl])
        for filename in os.listdir(path_images + folder):
            path_tmp = os.path.join(path_images, folder)
            image_filename = os.path.join(path_tmp, filename)
            img = read_image(image_filename).to(DEVICE)
            pred, _ = predict(img)
            if pred == i_cl:
                n_ok += 1
            n_tot += 1

        print('n_tot:', n_tot)
        print('n_ok:', n_ok)
        print(f'ok: {n_ok/n_tot:.1%}')


# %% load and plot image
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    filename = filedialog.askopenfilename()

    print('filename:', filename)
    if len(filename) == 0:
        print('\nPas de fichier sélectionné, fin du programme\n')
        quit()

    try:
        img = read_image(filename)
    except RuntimeError:
        print('\nCannot read file as image:', filename)
        print('Only jpeg and png are currently supported\n')
        quit()

    pred, argsort = predict(img)

    fig, ax = plt.subplots()
    ax.imshow(img.swapaxes(0, 2).swapaxes(0, 1).to('cpu'))
    ax.axis('off')
    _ = ax.set_title(
        f'race prédite : {classes[pred]}'
        + '\nautres races possibles:\n'
        + ', '.join([classes[i] for i in argsort[1:5]])
    )
    fig.tight_layout()
    plt.show()

###

# %% END OF FILE
###
