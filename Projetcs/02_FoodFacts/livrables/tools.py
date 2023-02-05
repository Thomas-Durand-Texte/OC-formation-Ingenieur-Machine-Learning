#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

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
            'savefig.transparent':True,
            'figure.dpi':150, # for better agreemet figsize vs real size
        }

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



# %% Analyse croisements des NaN

def croisement_NaN_counts( df_isna, keys ):
    # df_out = pd.DataFrame( { 'isna({:})'.format(key):np.zeros( len(keys), dtype=int) for key in keys },  index=['~isna({:})'.format(key) for key in keys] )
    df_out = pd.DataFrame( { key:np.zeros( len(keys), dtype=int) for key in keys },  index=[key for key in keys] )
    df_out.columns.name = '~isna \ isna'
    for i, keyi in enumerate( keys ):
        for j, keyj in enumerate( keys ):
            if i == j:
                df_out.iloc[i,j] = 0
                continue
            df_out.iloc[i,j] = ( ~df_isna[keyi] & df_isna[keyj] ).sum()
    return df_out


# %% Mesure distances

def distance_to_a_line( x, y, a, b ):
    ''' if a==0 : return (x-b)**2 '''
    if a == 0: return (x-b)**2
    # (dx,dy) = (1, a)
    # perpendicular : (-a, 1)
    # line perpendicular passing through x,y: y = x/(-a) + bp

    xp = ( y + x/a - b ) / (a + 1./a )

    return ( x - xp )**2 + ( a*xp + b - y )**2




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


# %% CHI-2

def chi2( data: pd.DataFrame, X: str, Y: str, normalised=True ):
    """ Perform a Chi-2 comparison

        X, Y: strings of keys used in data

        return xi_ji, contingence"""
    cont = data[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")
    tx = cont.loc[:,["Total"]]
    ty = cont.loc[["Total"],:]
    n = len(data)
    # indep = tx.dot(ty) / n  # Produit matriciel
    indep = tx @ (ty / n)

    contingences = cont.fillna(0).astype('int') # On remplace les valeurs nulles par 0
    xi_ij = (contingences-indep)**2/indep
    xi_n = xi_ij.sum().sum()
    if normalised:
        xi_ij = xi_ij/xi_n # normalisé de 0 à 1

    return xi_ij, contingences



# %% PCA

def correlation_graph(pca,
                      ij_F,
                      features,
                      ax=None) :
    """Affiche le graphe des correlations

    Positional arguments :
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    ij_F : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    ax : axis sur lequel le graphique est tracé (default None -> est créé)
    """

    # Extrait x et y
    x,y=ij_F

    # Taille de l'image (en inches)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 9))
    else:
        fig = ax.get_figure()

    # Pour chaque composante :
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0,
                pca.components_[x, i],
                pca.components_[y, i],
                head_width=0.07,
                head_length=0.07,
                width=0.02, )

        # Les labels
        ax.text(pca.components_[x, i] + 0.05*np.sign(pca.components_[x, i]),
                pca.components_[y, i] + 0.05*np.sign(pca.components_[y, i]),
                features[i])

    # Affichage des lignes horizontales et verticales
    ax.plot([-1, 1], [0, 0], color='grey', ls='--', zorder=0)
    ax.plot([0, 0], [-1, 1], color='grey', ls='--', zorder=0)

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    ax.set_ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # J'ai copié collé le code sans le lire
    ax.set_title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle
    an = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(an), np.sin(an), zorder=0 )  # Add a unit circle for scale

    # Axes et display
    ax.axis('equal')
    plt.show(block=False)
    return fig, ax


def distance_projection_on_plane_Fxy( X_scaled, X_proj, pca, x_y ):
    x,y = x_y
    X_proj_Fxy = X_proj[:,x].reshape(-1,1)*pca.components_[ x-1 ].reshape(1,-1) \
                + X_proj[:,y].reshape(-1,1)*pca.components_[ y-1 ].reshape(1,-1)
    # print( X_proj_Fxy.shape )
    # print( X_scaled.shape )
    return np.sqrt( ((X_proj_Fxy-X_scaled)**2).sum(1) )


def display_factorial_planes(   X_scaled,
                                x_y,
                                pca,
                                labels = None,
                                color=None,
                                alpha=1,
                                ax=None,
                                marker=".",
                                smin=5, smax=40 ):
    """
    Affiche la projection des individus

    DOCSTRING A METTRE A JOUR

    Positional arguments :
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments :
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = pca.transform(X_scaled)



    # On gère les labels
    if  labels is None :
        labels = []
    try :
        len(labels)
    except Exception as e :
        raise e

    # On vérifie la variable axis
    if not len(x_y) ==2 :
        raise AttributeError("2 axes sont demandées")
    if max(x_y )>= X_.shape[1] :
        raise AttributeError("la variable axis n'est pas bonne")

    # on définit x et y
    x, y = x_y

    # Initialisation de la figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7,6))
    else:
        fig = ax.get_figure()

    # # On vérifie s'il y a des clusters ou non
    # c = None if clusters is None else clusters


    dist = -distance_projection_on_plane_Fxy( X_scaled, X_, pca, x_y )

    
    dist = smax + ( (smin-smax) /dist.ptp()) * (dist-dist.min())

    # print('dist:', dist.shape, dist.dtype , 'n nan:', np.isnan(dist).sum())
    # print('X_:', X_.shape)
    # # print('c:', len(c))
    # print('x:', X_[:, x].shape)
    # print('y:', X_[:, y].shape)

    ax.scatter( X_[:,x], X_[:,y], s=dist, color=color )


    # # Les points
    # # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha,
    # #                     c=c, cmap="Set1", marker=marker)
    # # sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c, s=dist.ravel())
    # df = pd.DataFrame({'x':X_[:, x], 'y':X_[:, y], 'hue':c})
    # df['hue'] = df['hue'].astype('category')
    # # groups = df.groupby('hue')

    # # for name, group in groups:
    # #     ax.scatter(group['x'], group['y'], marker='o', linestyle='', s=dist, label=name)
    # #     ax.legend()


    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    # for group, clr in zip( df['hue'].cat.categories, colors):
    #     print("USE GROUPBY ?")
    #     sr_loc = df['hue'] == group
    #     df.loc[sr_loc,:].plot( kind='scatter', x='x', y='y',
    #                         s=dist[sr_loc.values], label=group, ax=ax,
    #                         color=clr )
    # lgnd = ax.legend()
    # for handle in lgnd.legendHandles:
    #     handle.set_markersize(6.0) # change markersize in legend

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
    if pca :
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else :
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    # ax.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    # ax.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) :
        # j'ai copié collé la fonction sans la lire
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            ax.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center')

    # Titre et display
    # ax.set_title(f"Projection des individus (sur F{x+1} et F{y+1})")
    ax.set_title(f"Projection des individus")
    # plt.show()





# %% END OF FILE
###
