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
from torchvision.transforms import Grayscale
from torchvision.io import read_image
import torch

import matplotlib.pyplot as plt
import utilities_tools_and_graph as graph_tools

import NeuralNetwork as nn
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


def chrono(sec):
    hours = int(sec // 3600)
    sec -= 3600 * hours
    mins = int(sec // 60)
    return f'{hours:02}:{mins:02}:{sec-60*mins:05.2f}'


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


def load_resize_save_images(infos: dict, path_load: str, path_save: str,
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
            image = read_image(os.path.join(path, filename))
            image = nn.__prepare_image__(image, target_shape)
            torch.save(image.to('cpu'),
                       os.path.join(path_save_tmp, filename[:-3]) + 'pt')
            n += 1
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
    df_annot_file_train.to_pickle(nn.ANNOTATION_FILENAME + '_train.pickle')
    df_annot_file_test.to_pickle(nn.ANNOTATION_FILENAME + '_test.pickle')
    Series(label_ind_to_str).to_pickle(nn.LABEL_IND_TO_STR_FILENAME)
    return


def display_annotation_files():
    df_annot_file_train = pd.read_pickle(nn.ANNOTATION_FILENAME
                                         + '_train.pickle')
    df_annot_file_test = pd.read_pickle(nn.ANNOTATION_FILENAME
                                        + '_test.pickle')
    print('train dataset:', df_annot_file_train.shape)
    print('test dataset:', df_annot_file_test.shape)
    print("Nombre total d'images:",
          len(df_annot_file_train) + len(df_annot_file_test))  # vérification
    display(df_annot_file_train.sample(10).style.set_caption('train files'))
    display(df_annot_file_test.sample(10).style.set_caption('test files'))


def test_class_dataset(dataset):
    image, label = dataset[10]
    # print('image type:', type(image), image.max())
    # print('label:', label, 'argmax:', label.argmax().item())
    label = dataset.get_label_str(label.argmax().item())
    image = image.swapaxes(0, 2).swapaxes(0, 1).to('cpu').numpy()
    fig, ax = plt.subplots(figsize=(5, 5*image.shape[0]/image.shape[1]))
    ax.imshow((image-image.min()) / image.ptp())
    ax.set_title(label)
    ax.grid(visible=False)
    ax.axis('off')
###


def define_data_loaders(train_data, test_data):
    train_dataloader = nn.DataLoader(train_data, batch_size=nn.BATCH_SIZE,
                                     shuffle=True)
    test_dataloader = nn.DataLoader(test_data, batch_size=nn.BATCH_SIZE,
                                    shuffle=True)
    return {'train': train_dataloader, 'test': test_dataloader}


def reload_trainer(trainer):
    dataloaders = trainer.dataloaders
    model = trainer.model
    mode = trainer.mode
    filename = 'tmp_reload'
    trainer.save_model(filename)
    del trainer
    trainer = nn.NN_Trainer(model, dataloaders, mode)
    trainer.load_model(filename)
    os.remove(os.path.join(PATH_MODELS + filename.replace(' ', '_')))
    return trainer


def load_trainer_results(filename: str):
    filename = os.path.join(PATH_RESULTS, filename)
    return load_pickle(filename)


# %% TOOLS AND PLOT FUNCS
def plot_examples_for_each_class(path_data, df, n_per_class):
    path_print = os.path.join(PATHS_PRINT['explore'], 'class_examples')
    greyscale = Grayscale()
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
                              logy_lr: bool = False,
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
    if logy_loss:
        axs[1].set_yscale('log')
    if logy_lr:
        ax_lr.set_yscale('log')
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
        axs[1].plot(x[1:], results_i['loss']['test'][1:], color=colors[i],
                    label=label)
        axs[1].plot(x[1:], results_i['loss']['train'][1:], '--',
                    color=colors[i])
        xmax = max(xmax, x[-1])

    if scheduler_info is not None:
        lrs = nn.get_scheduler_curve(optimizer_info, scheduler_info, x[-1])
        ax_lr.plot(lrs, '--r', label='scheduler')
        # with warnings.catch_warnings():
        #     ax_lr.relim()
        #     ax_lr.autoscale_view()
        #     warnings.filterwarnings("ignore", category=UserWarning)
        ax_lr.legend()
    
    if not logy_lr:
        ax_lr.set_ylim([0, ax_lr.get_ylim()[-1]])

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

    if len(results) > 1:
        axs[0].add_artist(lgd)
    fig.tight_layout()
    graph_tools.savefig(fig, PATHS_PRINT['essais'] + savename)


# def plot_schedulers():
#     lr_scheduler = torch.optim.lr_scheduler

#     def lmbda(epoch):
#         return 0.9 ** epoch

#     schedulers_tmp = {
#         'multiplicative': (lr_scheduler.MultiplicativeLR,
#                            dict(lr_lambda=lmbda)),
#         'lambda': (lr_scheduler.LambdaLR,
#                    dict(lr_lambda=lmbda)),
#         'step': (lr_scheduler.StepLR,
#                  dict(step_size=10, gamma=0.5)),
#         'multi-step': (lr_scheduler.MultiStepLR,
#                        dict(milestones=[30, 50, 70], gamma=0.4)),
#         'exponential': (lr_scheduler.ExponentialLR,
#                         dict(gamma=0.95)),
#         'cosine annealing': (lr_scheduler.CosineAnnealingLR,
#                              dict(T_max=30, eta_min=0)),
#         'cyclic triangular': (lr_scheduler.CyclicLR,
#                               dict(base_lr=0.001, max_lr=100,
#                                    step_size_up=10, mode="triangular")),
#         'cyclic triangular 2': (lr_scheduler.CyclicLR,
#                                 dict(base_lr=0.001, max_lr=100,
#                                      step_size_up=10, mode="triangular2")),
#         'cyclic exp range': (lr_scheduler.CyclicLR,
#                              dict(base_lr=0.001, max_lr=100, step_size_up=10,
#                                   mode="exp_range", gamma=0.95)),
#         'one cycle cos': (lr_scheduler.OneCycleLR,
#                           dict(max_lr=100, steps_per_epoch=10, epochs=10)),
#         'one cycle linear': (lr_scheduler.OneCycleLR,
#                              dict(max_lr=100, steps_per_epoch=10, epochs=10,
#                                   anneal_strategy='linear')),
#         'cosine annealing warm restart':
#             (lr_scheduler.CosineAnnealingWarmRestarts,
#              dict(T_0=30, T_mult=1, eta_min=0.001, last_epoch=-1))
#     }

#     names_groups = [['lambda', 'multiplicative', 'exponential'],
#                     ['step', 'multi-step'],
#                     ['cosine annealing', 'cosine annealing warm restart'],
#                     ['cyclic triangular', 'cyclic triangular 2',
#                      'cyclic exp range'],
#                     ['one cycle cos', 'one cycle linear']
#                     ]

#     for i, names in enumerate(names_groups):
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.set_xlabel('epochs')
#         ax.set_ylabel('learning rate')

#         for name in names:
#             model = torch.nn.Linear(2, 1)
#             optimizer = torch.optim.SGD(model.parameters(), lr=100)
#             scheduler, params = schedulers_tmp[name]
#             scheduler = scheduler(optimizer, **params)
#             lrs = []
#             for ii in range(100):
#                 optimizer.step()
#                 lrs.append(optimizer.param_groups[0]["lr"])
#                 scheduler.step()
#             ax.plot(range(100), lrs, label=name)

#         ax.legend()
#         fig.tight_layout()
#         graph_tools.savefig(fig, PATHS_PRINT['essais']
#                                  + f'schedulers_tmp/{i}')


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

    fig.update_layout(width=800, height=600)
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


def plot_activations():
    x = torch.linspace(-6, 2, 100)
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, (func, label, linestyle) in enumerate([
                    (nn.nn.ReLU(), 'ReLU', '-'),
                    (nn.nn.LeakyReLU(), 'Leaky ReLU', '--'),
                    (nn.nn.ELU(), 'ELU', ':'),
                    (nn.nn.SiLU(), 'SiLU', '-'),
                    (nn.nn.Mish(), 'Mish', '-'),
                      ]):
        y = func(x)
        ax.plot(x.numpy(), y.numpy(), linestyle=linestyle, label=label,
                linewidth=3-0.5*i)
    ax.legend('entrée')
    ax.legend('sortie')
    ax.legend()
    fig.tight_layout()
    graph_tools.savefig(fig, PATHS_PRINT['essais'] + 'activations')

# %% END OF FILE
###
