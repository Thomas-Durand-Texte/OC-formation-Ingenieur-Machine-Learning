#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% IMPORT PACKAGES
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pickle
import warnings

from sklearn import model_selection
from torchvision.io import read_image
import torch

import matplotlib.pyplot as plt
import utilities_tools_and_graph as graph_tools

# import NeuralNetwork as nn
from common_vars import PATH_MODELS, PATH_RESULTS, PATHS_PRINT


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


def make_folder_from_file(filename):
    make_folder(filename[:filename.rindex(os.sep)])


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


def load_trainer_results(filename: str):
    filename = os.path.join(PATH_RESULTS, filename)
    return load_pickle(filename)


def chrono(sec):
    hours = int(sec // 3600)
    sec -= 3600 * hours
    mins = int(sec // 60)
    return f'{hours:02}:{mins:02}:{sec-60*mins:05.2f}'


# %%
def plot_filtres_first_layer(weights, pattern_per_row, figsize):
    # print('weights:', weights.shape)
    n = 0
    for i in range(weights.shape[0]//pattern_per_row):
        fig, axs = plt.subplots(figsize=figsize, ncols=pattern_per_row)
        for j in range(pattern_per_row):
            pattern = weights[n].detach().clone()
            # pattern -= pattern.min()
            # pattern /= pattern.max()
            # print('pattern:', pattern.shape)
            # print('pattern:', pattern.shape)

            vmax = torch.abs(pattern).max()
            axs[j].imshow(
                torch.cat((pattern[0], pattern[1])),
                cmap='gray', vmin=-vmax, vmax=vmax
            )
            axs[j].axis('off')
            n += 1
        fig.tight_layout()

# %% END OF FILE
###
