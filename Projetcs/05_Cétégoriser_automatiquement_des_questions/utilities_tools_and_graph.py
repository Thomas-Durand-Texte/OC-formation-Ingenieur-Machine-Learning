#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions used, notably for graphic parameters
TODO: list of functions and classes

Classes:

    xx
    xxx

Functions:

    set_theme(bool)
    make_folder(path_folder:str)
    concat_folders(*args) -> str
    ...

Misc variables:

    __version__
    format_version
    compatible_formats
"""

import os
import subprocess

import seaborn as sns


def set_theme(white_font=True):
    """Réglages graphiques

    Args:
        white_font (bool, optional): set if font a set to white or black.
        Defaults to True.
    """
    if white_font:
        # base_color, grey, blck = '0.84' , '0.5', 'k'
        base_color = '0.84'
    else:
        # base_color, grey, blck = 'k', '0.5', '0.84'
        base_color = 'k'
    rc_params = {
        'figure.facecolor': (0.118,)*3,
        'axes.labelcolor': base_color,
        'axes.edgecolor': base_color,
        'axes.facecolor': (0, 0, 0, 0),
        'text.color': 'white',
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage[cm]{sfmath} \usepackage{amsmath}',
        'font.family': 'sans-serif',
        'font.sans-serif': 'DejaVu Sans',
        'xtick.color': base_color,
        'ytick.color': base_color,
        "axes.grid": True,
        "grid.color": (0.5,)*3,
        "grid.linewidth": 0.35,
        "grid.linestyle": (7, 10),  # plain, space
        'legend.edgecolor': '0.2',
        'legend.facecolor': (0.2, 0.2, 0.2, 0.6),
        # 'legend.framealpha':'0.6',
        'pdf.fonttype': 42,
        'savefig.format': 'pdf',
        'savefig.transparent': True,
        'figure.dpi': 150,  # for better agreemet figsize vs real size
        }

    sns.set_theme('notebook', rc=rc_params)


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


def concat_folders(*args) -> str:
    """Concatenate strings to form a path to a folder

    exemple concat_filders('/abc', 'ABC') -> '/abc/ABC/'

    Returns:
        str: concatenated string corresponding to a path to a folder
    """
    str_path = ''
    for arg in args:
        if arg == '..':
            str_path = str_path[:str_path[:-1].rfind(os.sep)+1]
        else:
            str_path += arg
        if str_path[-1] != os.sep:
            str_path += os.sep
    return str_path


class Path():
    """ Path(s_in='', s_lim=None)
        create a path to the string s_in (default is current path)
        and stops after s_lim """
    n_Path = 0

    def __init__(self, s_in='', s_lim=None):
        """Initialisation of class Path

        Args:
            s_in (str, optional): string corresponding to a folder path.
            Defaults to '' (current working directory).
            s_lim (_type_, optional): chain to limit the path. Defaults to None
            Example s_in = '/folder_1/folder_2/folder_3', s_lim = '2', leads to
            '/folder_1/folder_2/'
        """
        if s_in == '':
            s_in = os.getcwd()
        if s_lim is not None:
            if s_lim in s_in:
                s_in = s_in[:s_in.index(s_lim) + len(s_lim)]
        self.str_path = concat_folders(s_in)
        Path.n_Path += 1
        return

    def __add__(self, other) -> str:
        """ Path + str : return str """
        return self.str_path + str(other)

    def __truediv__(self, other):
        """ Path / str : return path concatenated"""
        return Path(concat_folders(self.str_path, str(other)))

    def __invert__(self) -> str:
        """ ~Path : return str of the path """
        return self.str_path

    def __str__(self) -> str:
        """ __str__ return str of the path """
        return self.str_path
    # __str__ #

    def makedir(self):
        """ Create a folder from current path if it does not exists """
        return make_folder(self)


def gs_opt(filename: str):
    """otpimisation of a pdf file with gosthscript

    Args:
        filename (str): string corresponding to the pdf file to optimise
    """
    filename_tmp = filename.replace('.pdf', '') + '_tmp.pdf'
    command = ['gs',
               '-sDEVICE=pdfwrite',
               '-dEmbedAllFonts=true',
               '-dSubsetFonts=true',  # Create font subsets (default)
               '-dPDFSETTINGS=/prepress',  # Image resolution
               # Embeds images used multiple times only once
               '-dDetectDuplicateImages=true',
               # Compress fonts in the output (default)
               '-dCompressFonts=true',
               '-dNOPAUSE',  # No pause after each image
               '-dQUIET',  # Suppress output
               '-dBATCH',  # Automatically exit
               '-sOutputFile='+filename_tmp,  # Save to temporary output
               filename]  # Input file

    # Create temporary file
    subprocess.run(command, check=True)
    # Delete input file
    subprocess.run('rm -f ' + filename, shell=True, check=True)
    # Rename temporary to input file
    subprocess.run('mv -f ' + filename_tmp + " " + filename,
                   shell=True, check=True)


def savefig(fig, savename: str, **kwargs):
    """Saves a figure with kwargs (fig.savefig(savename, **kwargs)).
       A check is done first to determine if a folder has to be created
       according to savename.
       Finally, if the file is saved as .pdf, gosthscript optimisation is
       performed.

    Args:
        fig : matplotlib figure
        savename (str): string of the name of the output pdf file
    """
    if os.sep in savename:
        make_folder(savename[:savename.rindex(os.sep)])
    fig.savefig(savename, **kwargs)
    savename += '.pdf'
    if os.path.isfile(savename):
        gs_opt(savename)
    return


def image_size_from_shape(shape: tuple, width: float = None,
                          height: float = None, ymargin=0.):
    """Compute tuple (width, height) from shape tuple of an image an width
        or height

    Args:
        shape (tuple): shape of the image (n_rows, n_cols)
        width (float, optional): desired width. Defaults to None.
        height (float, optional): desired height. Defaults to None.
        ymargin (_type_, optional): margin added to the output height.
                                    Defaults to 0..

    Returns:
        tuple: (width, height)
    """
    if width is not None:
        return width, width*shape[0]/shape[1] + ymargin
    if height is not None:
        return height*shape[1]/shape[0], height
    print('\n!!! image_size_from_shape: '
          'nor width nor heigth was provided !!!\n')
    return shape


set_theme()
del set_theme


def df2ltx(data, keys_replace_columns=None, col_format='c',
           keys_replace_indexes=None) -> str:
    """return string corresponding to latex version of the givend DataFrame.

    Args:
        data (pandas.DataFrame): DataFrame to export to latex. Only single
        indexing is currently supported.
        keys_replace_columns (dict, optional): dict for caracter remapping for
                                               columns. Defaults to {'_': ' '}.
        col_format (str, optional): latex format for columns. Defaults to 'c'.
        keys_replace_indexes (dict, optional): dict for caracter remapping for
                                               indexes. Defaults to {'_': ' '}.

    Returns:
        str: caracter chain for latex
    """
    if keys_replace_columns is None:
        keys_replace_columns = {'_': ' '}
    if keys_replace_indexes is None:
        keys_replace_indexes = {'_': ' '}
    _, n_keys = data.shape
    strng = '\\begin{tabular}{' + f'{col_format*(n_keys+1)}' + '}\n\t'
    strng += '\\rowcolor{dfFirstRow}\n\t'
    for col in data.columns:
        for key, val in keys_replace_columns.items():
            col = col.replace(key, val)
        strng += ' & ' + col
    i = 1
    for index, row in zip(data.index, data.values):
        index = str(index)
        for key, val in keys_replace_indexes.items():
            index = index.replace(key, val)
        i = (i+1) % 2
        strng += ' \\tabularnewline\n\t\\rowcolor{' \
                 + ['dfEvenRow', 'dfOddRow'][i] \
                 + '}\n\t' + str(index)
        for cell in row:
            cell = str(cell)
            if cell == 'nan':
                cell = 'NaN'
            strng += ' & ' + cell

    strng += '\n\\end{tabular}'
    return strng

###


# %% markdown

###


# %%

###


# %% END OF FILE
###
