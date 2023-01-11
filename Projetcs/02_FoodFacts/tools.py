#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import os
import subprocess

cm = 1./2.54

# %% Basic and plot tools
def set_theme( white_font=True ):
    """ set_theme( white_font=True ) """
    if white_font: wht, grey, blck = '0.84' , '0.5', 'k'
    else: wht, grey, blck = 'k', '0.5', '0.84'
    rc = { 'figure.facecolor':(0.118,)*3, 
            'axes.labelcolor':wht,
            'axes.edgecolor':wht,
            'axes.facecolor':(0,0,0,0),
            'text.color':'white',
            'text.usetex':False,
            'text.latex.preamble':r'\usepackage[cm]{sfmath} \usepackage{amsmath}' ,
            'font.family': 'sans-serif' ,
            'font.sans-serif': 'DejaVu Sans' ,
            'xtick.color':wht,
            'ytick.color':wht,
            "axes.grid" : True,
            "grid.color": (0.7,)*3,
            "grid.linewidth": 0.4,
            "grid.linestyle": (10,5),
            'legend.edgecolor':'0.2',
            'legend.facecolor':(0.2,0.2,0.2,0.6),
            # 'legend.framealpha':'0.6',
            'pdf.fonttype':42,
            'savefig.format':'pdf',
            'savefig.transparent':True }
 
    sns.set_theme( 'notebook' , rc=rc )
    return



def make_folder( path_folder ):
    path_folder = path_folder.__str__()
    try:
        if os.path.isdir( path_folder ) : return
        os.makedirs(path_folder)
    except OSError:
        pass
    return

def concat_folders(*args, **kwargs):
    """ concat_folders(*args, **kwargs)
        concatenate folders in args (strings) """
    sPath = ''
    for arg in args:
        if arg == '..': sPath = sPath[:sPath[:-1].rfind(os.sep)+1]
        else: sPath += arg
        if sPath[-1] != os.sep: sPath += os.sep
    return sPath

class Path(object):
    """ Path( s_in='', s_lim=None)
        create a path to the string s_in (default is current path)
        and stops after s_lim """
    n_Path = 0
    def __init__(self, s_in='', s_lim=None):
        """docstring."""
        if s_in == '': s_in = os.getcwd()
        if not s_lim is None:
            if s_lim in s_in:
                s_in = s_in[ :s_in.index( s_lim ) + len(s_lim) ]
        self.sPath = concat_folders(s_in)
        self.N = Path.n_Path
        Path.n_Path += 1

    def __add__(self, other):
        """ Path + str : return str """
        if isinstance(other, str): return self.sPath + other

    def __truediv__(self, other):
        """ Path / str : return path concatenated"""
        if isinstance(other, str): return Path(concat_folders(self.sPath, other))

    def __invert__(self):
        """ ~Path : return str of the path """
        return self.sPath

    def __str__(self):
        """ __str__ return str of the path """
        return self.sPath
    # __str__ #

    def makedir( self ):
        return make_folder( self )


def gs_opt( filename ):
    """ otpimisation of a pdf file with gosthscript """
    filenameTmp = filename.replace('.pdf', '') + '_tmp.pdf'
    gs = ['gs',
            '-sDEVICE=pdfwrite',
            '-dEmbedAllFonts=true',
            '-dSubsetFonts=true',             # Create font subsets (default)
            '-dPDFSETTINGS=/prepress',        # Image resolution
            '-dDetectDuplicateImages=true',   # Embeds images used multiple times only once
            '-dCompressFonts=true',           # Compress fonts in the output (default)
            '-dNOPAUSE',                      # No pause after each image
            '-dQUIET',                        # Suppress output
            '-dBATCH',                        # Automatically exit
            '-sOutputFile='+filenameTmp,      # Save to temporary output
            filename]                         # Input file

    subprocess.run(gs)                                      # Create temporary file
    subprocess.run( 'rm -f ' + filename, shell=True)            # Delete input file
    subprocess.run( 'mv -f ' + filenameTmp + " " + filename, shell=True) # Rename temporary to input file

def savefig( fig, savename, **kwargs ):
    """ savefig( fig, savename, **kwargs ) 
        Saves a figure with kwargs (fig.savefig( savename, **kwargs) ).
        A check is done first to determine if a folder has to be created according to savename.
        Finally, if the file is saved as .pdf, gosthscript optimisation is performed. """
    if os.sep in savename: make_folder( savename[:savename.rindex(os.sep)] )
    fig.savefig( savename, **kwargs )
    savename += '.pdf'
    if os.path.isfile( savename ): gs_opt( savename )


def plot_test_figure():
    import numpy as np
    x = np.linspace( 0, 10, 100 )
    y = np.sin( x )

    fig, ax = plt.subplots( figsize=(15*cm,12*cm) )
    ax.plot( x, y, 'r', label='sine' )
    ax.legend()
    plt.show()


# %% string managament


def value_count_labels_in_string_series( series, split_string=',' ):
    """ dico_out = ount_labels_in_string_series( series ) 
        return a dictionnary like {label_1:n1, label_2:n2, ...} """
    out = {}
    for string in series.to_list():
        try: string.split(split_string)
        except: 
            print("error: '{:}' is not a string".format(string), type(string))
            return
        for label in string.split(split_string):
            if label in out : out[label] += 1
            else: out[label] = 1
    return out



if False: # Test extract and count labels
    import pandas as pd
    sr = pd.Series( ['a,b,c', 'd,e,a', 'b,c,e', 'f,g,d'])
    sr.to_list
    out = count_labels_in_string_series( sr )
    print('out', out)


# %% categories mangement

def elems_containing_keys( elems, keys):
    """ values_bool = lst_elem_contain_key( elems, keys) """
    return pd.Series( elems ).str.contains( '|'.join(keys) , regex=False ).values

def lst_str_remove_items_containing_key( lst_str, lst_keys_to_remove ):
    """ lst_out = lst_str_remove_items_containing_key( lst_str, lst_keys_to_remove )\n\n 
        Create a new list with the items (strings) not containing one of the key in lst_keys_to_remove
        """
    lst_out = []
    for cat in lst_str:
        not_to_add = False
        for key in lst_keys_to_remove: 
            if key in cat:
                not_to_add = True
                break
        if not_to_add: continue
        lst_out.append( cat ) 
    return lst_out

def lst_str_keep_items_containing_key( lst_str , lst_keys_to_keep ):
    """ lst_out = lst_str_keep_items_containing_key( lst_str, lst_keys_to_keep )\n\n
        Create a new list with the items (strings) containing one of the key in lst_keys_to_keep
        """
    lst_out = []
    for cat in lst_str:
        to_add = False
        for key in lst_keys_to_keep: 
            if key in cat:
                to_add = True
                break
        if to_add:
            lst_out.append( cat ) 
    return lst_out




# %% END OF FILE
###
