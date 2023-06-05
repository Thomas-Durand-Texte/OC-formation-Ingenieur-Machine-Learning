# -*- coding: utf-8 -*-
"""_summary_
"""
# %% Import des packages
import os
import importlib
import time

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray
# import scipy.stats as st
from sklearn import model_selection, metrics, preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import manifold
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cross_decomposition import CCA

from gensim.models import KeyedVectors

from sentence_transformers import SentenceTransformer

import tensorflow as tf
import tensorflow_hub as hub


import pyLDAvis.lda_model

from sklearn.pipeline import Pipeline

import nltk
import contractions
import re
from string import punctuation

import matplotlib.pyplot as plt
import seaborn as sns

import html_tools
import utilities_tools_and_graph as graph_tools

pyLDAvis.enable_notebook()
pd.set_option("display.max_columns", 200)
PATH_PRINT = 'Figures/'
contractions.add('app', 'application')


# %% Affichage de l'arborescence
def print_listdir(path: str = None,
                  level: int = 0,
                  exclude: list = None):
    """Affiche l'arborescence des dossiers et fichiers

    Args:
        path (str, optional): chemin du dossier à explorer.
        level (int, optional): niveau d'indentation. Defaults to 0.
        exclude (list, optional): dossiers = exclure. Defaults to [].
    """
    if exclude is None:
        exclude = []
    suffix = ''
    if level > 0:
        suffix = ' |-' * level
    vals = os.listdir(path)
    vals.sort()
    if path is None:
        path = ''
    for val in vals:
        if val in exclude:
            continue
        print(suffix, val)
        if os.path.isdir(path + val):
            print_listdir(path + val + '/', level+1)


def load_data():
    """load data

    Returns:
        DataFrame: loaded Data
    """
    path = 'data/source/'
    filename = ''

    lst_files = os.listdir(path)
    data = pd.concat((pd.read_csv(path+filename)
                      for filename in lst_files))
    data['CreationDate'] = pd.to_datetime(data['CreationDate'])

    del lst_files, filename

    print(data.shape)
    data.head()
    return data


def plot_ratio_isnull(df: DataFrame):
    """draw and save a barplot corresponding to missing values per column

    Args:
        df (DataFrame): input data
    """
    ax = (df.isnull().mean(axis=0)*100).plot.barh()
    ax.set_title("Ratio des valeurs manquantes par colonne")
    ax.set_xlabel('ratio (%)')
    fig = ax.get_figure()
    fig.set_size_inches(5, 3)
    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'ratio_isnull.pdf')


def length_per_year(sr_in: Series, years: Series):
    """Compute an array with length of values for a variable in a
        DataFrame for different years

    Args:
        sr_in (Series): input data
        year (Series): years of each element in sr_in

    Returns:
        array: (n_year, n_bins)
    """
    year_min = years.min()
    year_max = years.max()
    sr_len = sr_in.apply(lambda x: len(x))
    bins = np.linspace(sr_len.min(), sr_len.max()+1, 31)
    out = np.zeros((year_max-year_min+1, bins.size-1))
    for year, out_i in zip(range(year_min, year_max+1), out):
        hist, _ = np.histogram(sr_len.loc[years == year], bins)
        out_i[:] = hist[:] / hist.sum()
    return (out, bins, year_min, year_max)


def bins_centers(bins: np.ndarray):
    """Return array of center of bins

    Args:
        bins (array): bins provided by np.histrogram for example

    Returns:
        array: center of bins = 0.5*(bin[i]+bins[i+1])
    """
    return np.array([int(0.5*(bins[i]+bins[i+1])+0.5)
                     for i in range(0, bins.size-1)])


def plot_array_len_per_year(hist: np.ndarray, bins, year_min, year_max,
                            varname, log=False):
    """plot an histogram (years x bins) corresponding to the output
       of length_per_year function

    Args:
        hist (array): histogram data (years x bins)
        bins (array): bins for the histogram
        varname (str): Name for the variable
        log (bool, optional): wheter to plot log(hist) or not. Defaults False.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.grid(visible=False)
    if log:
        log_hist = np.log(hist+1e-12)
        im = ax.imshow(log_hist, cmap='hot', vmin=np.log(0.001))
    else:
        im = ax.imshow(100*hist, cmap='hot')
    ax.set_aspect('auto')
    fig.colorbar(im, shrink=0.9,
                 label='proportion ({:})'.format('log' if log else '%'))
    ax.set_ylabel('année')
    ax.set_xlabel(f'longueur (caractères)')
    ax.set_title('Évolution temporelle de la longueur de la variable'
                 + f' "{varname}"'
                 )

    ax.set_yticks(np.arange(hist.shape[0]))
    ax.set_yticklabels(np.arange(year_min, year_max+1))

    step_x = 4
    ax.set_xticks(np.arange(0, hist.shape[1], step_x))
    ax.set_xticklabels(bins_centers(bins)[::step_x])

    fig.tight_layout()
    graph_tools.savefig(fig,
                        PATH_PRINT + f'arr_length_per_year_{varname.lower()}')


def plot_histogram_longueur_texte(sr_body):
    hist, bins = np.histogram(sr_body.apply(lambda x: len(x)), bins=31)
    val_limit = 5
    hist_prop = hist * (100/len(sr_body))
    b_low = hist_prop < val_limit
    nmax = int(bins[np.arange(hist.size)[b_low][0]] + 0.5)

    x = bins_centers(bins)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hlines(val_limit, bins.min(), bins.max(), color='r', linestyle='--',
              label='proportion limite')
    ax.annotate(f'{val_limit}%', [15000, val_limit-1.2], color='r',
                ha='center', va='top')

    ax.vlines(nmax, 0.01, 50, color='y', linestyle='--')
    ax.annotate(f'{nmax}\ncatactères', [nmax, 0.003], color='y',
                ha='center', va='center')

    ax.semilogy(x, hist_prop, '-o')
    ax.plot(x[b_low], hist_prop[b_low], '-o', label='longueurs rejetées')
    ax.legend()
    ax.set_ylabel('proportion (%)')
    ax.set_xlabel('longueur (catactères)')
    ax.set_title('Histogramme de la longueur des corps de texte des questions')
    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'hist_text_length')


def init_tag(sr_tags: Series) -> Series:
    """generate list of tags from initial string provided by stackexchange

    Args:
        sr_tags (Series): strings of tags

    Returns:
        Series: list of list of tags
    """
    mapping = {'><': ',',
               '<|>': ''}
    return sr_tags.replace(mapping, regex=True).str.split(',')


def count_tag(sr: Series) -> Series:
    """count the number of tags in a given Series

    Args:
        sr (Series): input Series of list of Tags

    Returns:
        Series: ouput Series with the number of occurence per Tag
    """
    # tags_freqDist = nltk.FreqDist([tag for tags in sr for tag in tags])
    # print(tags_freqDist)
    tags = {}
    for tags_i in sr:
        for tag in tags_i:
            if tag not in tags:
                tags[tag] = 1
                continue
            tags[tag] += 1
    return pd.Series(tags)


def count_tags_and_questions_per_year(df: pd.DataFrame) -> tuple:
    """count the number of tags and questions year by year
    in the DataFrame

    Args:
        df (DataFrame): input data

    Returns:
        tuple: (tags_per_year (dict), questions_per_year (Series))
    """
    questions_per_year = {}
    tags_per_year = {}
    years = df['CreationDate'].dt.year.unique()
    years.sort()
    for year in years:
        loc = df['CreationDate'].dt.year == year
        tags_per_year[year] = count_tag(df.loc[loc, 'Tags'])
        questions_per_year[year] = loc.sum()

    return tags_per_year, pd.Series(questions_per_year)


def plot_questions_per_year(q_per_year: Series):
    """plot histogram corresponding to the number of questions per year

    Args:
        q_per_year (Series): out of function
            count_tags_and_questions_per_year
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    q_per_year.plot(ax=ax)
    ax.set_xlabel('année')
    ax.set_ylabel('nombre de questions')
    ax.set_title('Évolution du nombre de questions par année')
    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'q_per_y')


def plot_tags_most_present(tags_per_year: dict):
    """plot bar plot of most used tags in 2011 and 2022

    Args:
        tags_per_year (dict): out of function count_tags_and_questions_per_year
    """
    fig, axs = plt.subplots(nrows=2, figsize=(7, 7))
    for ax, year in zip(axs, [2011, 2022]):
        ax.set_title(year)
        ax.set_ylabel("nombre d'occurence")
        sr_plot = tags_per_year[year].sort_values(ascending=False).iloc[:30]
        sr_plot.plot.bar(ax=ax)

    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'tags_per_y')


def compute_most_used_tags(tags: dict,
                           n_tags: int,
                           year_min: int,
                           ) -> list:
    """count all tags and compute most used ones for creation_year > year_min.
    the dict tags is updated with keys 'full' (Series of tags ratio) and
    'most used' (Series)

    Args:
        tags (dict): storage of data (containing key 'per year')
        n_tags (int): number of tags to keep
        year_min (int): year to start to count
    Returns:
        list: list of years (int)
    """
    tags_full = None
    for year, sr_tags in tags['per year'].items():
        if year_min > year:
            continue
        if tags_full is None:
            tags_full = sr_tags.copy() / sr_tags.sum()
            years = [year]
            continue
        # tags_full.add(sr_tags, fill_value=0)
        tags_full.add(sr_tags / sr_tags.sum(), fill_value=0)
        years.append(year)
        # tags_full.combine( sr_tags / sr_tags.sum(), max )

    tags_full = 100 * (tags_full/len(years))
    # tags_full = tags_full.astype('int')
    most_used_tags = tags_full.sort_values(ascending=False).iloc[:n_tags]
    tags['full'] = tags_full
    tags['most used'] = most_used_tags
    return years


def plot_most_used_tags(tags: dict):
    """plot the barplot of the most used tags

    Args:
        tags (dict): storage dict updated with compute_most_used_tags
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    tags['most used'].plot.bar(ax=ax)

    ax.set_ylabel('ratio annuel moyen (%)')
    ax.set_title('Ratio annuel moyen des tags les plus fréquents')
    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'most_used_tags')


def plot_propotion_most_used_tags_per_year(tags_per_year: dict,
                                           selected_tags: list,
                                           years: list):
    array_out = np.empty((len(years), len(selected_tags)))
    for i, year in enumerate(years):
        tags = tags_per_year[year]
        array_out[i, :] = tags[selected_tags] * (100/tags.sum())

    inds_sort = array_out[-1].argsort()[::-1]
    array_out = array_out[:, inds_sort]
    selected_tags = Series(selected_tags)[inds_sort]

    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(array_out, cmap='hot')
    plt.colorbar(im, shrink=0.9, label='propotion (%)')
    ax.grid(visible=False)  # remove grid
    ax.set_aspect('auto')

    ax.set_title("Proportion d'utilisation des tags par année")

    ax.set_ylabel('année')
    ax.set_yticks([i for i in range(len(years))])
    ax.set_yticklabels([year for year in years])

    ax.set_xticks([i for i in range(selected_tags.size)])
    ax.set_xticklabels([tag for tag in selected_tags], rotation=90)
    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'most_used_tags_per_year')


def plot_tags_dataset(df_tags: DataFrame):
    """plot the proportion of tags in the given dataset

    Args:
        df_tags (DataFrame): computed with compute_y_tag()
    """
    y = df_tags.sum(axis=0) * (100 / len(df_tags))
    # print(y)
    fig, ax = plt.subplots(figsize=(5, 3))
    y.plot.bar(ax=ax)
    ax.set_title('Proportion de questions contenant chaque tag')
    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'prop_tags.pdf')


def filter_tags(tags_in: list, tags_selected: list):
    """for a given list of tags, return tags present in selected ones

    Args:
        tags_in (list): input tags
        tags_selected (list): filtering tags

    Returns:
        list / np.nan: filtered tags
    """
    out = [tag for tag in tags_in if tag in tags_selected]
    if len(out) == 0:
        return np.nan
    return out


def apply_tag_filtering(df: DataFrame,
                        years: list,
                        most_used_tags: list) -> DataFrame:
    """Create a new DataFrame and keep only tags within the provided list.
    The rows/questions without any tag are removed.

    Args:
        df (DataFrame): input data
        years (list): list of years (to print informations)
        most_used_tags (list): tags to be kept

    Returns:
        DataFrame: filtered DataFrame
    """
    print(f'Total number of questions: {len(df)}')
    loc = df['CreationDate'].dt.year >= min(years)
    print(f'Number of questions since {min(years)}: {loc.sum()}')

    out = df.copy()
    out['Tags'] = out['Tags'].apply(filter_tags, args=(most_used_tags,))

    vars = ['Title', 'Tags']
    display(df[vars].head().style.set_caption('original'))
    display(out[vars].head().style.set_caption('filtered'))
    out = out.loc[~out['Tags'].isna(), :]
    display(out[vars].head().style.set_caption('tag filtered and empty list'
                                               ' dropped'))

    print(f'Total number of questions with Tags filtered: {len(out)}')
    loc = out['CreationDate'].dt.year >= min(years)
    print(f'Number of questions with Tags filtered since {min(years)}: '
          f'{loc.sum()}')
    return out


def get_html_tags(sr_body) -> set:
    if True:  # exlore data to check all the html tags
        html_tags = set()
        for i, body in enumerate(sr_body):
            print(f'{i/len(sr_body):.1%}', end='\r')
            html_tags.update(html_tools.get_different_tags(body))
        print(f'{1:.0%} ')
    elif False:
        html_tags = {'blockquote', 'code', 'p', 'pre'}
    else:
        html_tags = re.compile(
                            r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return html_tags


def html_tags_remapping(html_tags: list) -> dict:
    return '|'.join([f'<{tag}>' for tag in html_tags]
                    + [f'</{tag}>' for tag in html_tags])


# source notebook nlp-preprocessing-feature-extraction-methods-a-z
# LONG NG - Kaggle
def remove_html(text):
    """
        Remove the html in sample text
    """
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)


# source notebook nlp-preprocessing-feature-extraction-methods-a-z
# LONG NG - Kaggle
def remove_URL(text):
    """
        Remove URLs from a sample string
    """
    return re.sub(r"https?://\S+|www\.\S+", "", text)


# source notebook nlp-preprocessing-feature-extraction-methods-a-z
# LONG NG - Kaggle
def remove_non_ascii(text):
    """
        Remove non-ASCII characters
    """
    # or ''.join([x for x in text if x in string.printable])
    return re.sub(r'[^\x00-\x7f]', r'', text)


# source notebook nlp-preprocessing-feature-extraction-methods-a-z
# LONG NG - Kaggle
EMOJI_PATTERN = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)


def remove_special_characters(text):
    """
        Remove special special characters, including symbols, emojis,
        and other graphic characters
    """
    return EMOJI_PATTERN.sub(r'', text)


def clean_body(sr_body: Series) -> Series:
    # lower all strings
    sr_body = sr_body.str.lower()
    # remove html tags
    # sr_body = sr_body.str.replace(html_tags_remapping(get_html_tags(sr_body))
    #                               ' ', regex=True)
    for func in [remove_URL,
                 remove_html,
                 remove_non_ascii,
                 contractions.fix,  # remove text contractions
                 remove_special_characters
                 ]:
        sr_body = sr_body.apply(func)
    return sr_body


def clean_title(sr_title: Series) -> Series:
    # lower all strings
    sr_title = sr_title.str.lower()
    for func in [remove_URL,
                 contractions.fix,  # remove text contractions
                 remove_special_characters
                 ]:
        sr_title = sr_title.apply(func)
    return sr_title


def concat_str_features(data, features):
    return data.apply(lambda x: ' '.join([x[feature] for feature in features]),
                      axis=1)


def replace_kewords(text):
    text = text.replace('c#', 'csharp')
    return text


SW = set(tuple(nltk.corpus.stopwords.words('english')))


def tokenize(text, tags):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # tokenizer = nltk.tokenize.word_tokenize
    # tokenizer = nltk.RegexpTokenizer(r'\w+').tokenize
    out = []
    regex = re.compile('[' + re.escape(punctuation) + ']')
    for word in tokenizer.tokenize(text):
        if (word in SW) or (word in punctuation):
            continue
        if word in tags:
            out.append(word)
            continue
        out.append(regex.sub('', word))
    return out


def pos_tag_to_wordnet(pos_tag):
    pos_tag = pos_tag[:2]
    if pos_tag == 'NN':
        return 'n'
    if pos_tag == 'JJ':
        return 'a'
    if pos_tag == 'RB':
        return 'r'
    if pos_tag == 'VB':
        return 'v'
    return 'n'


def lemmatize(words: list, tags: list) -> list:
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_postag = nltk.pos_tag(words)
    return [word if word in tags
            else lemmatizer.lemmatize(word, pos=pos_tag_to_wordnet(pos_tag))
            for word, pos_tag in w_postag]


def stem(words: list, tags: list) -> list:
    ps = nltk.stem.PorterStemmer()
    return [word if word in tags else ps.stem(word) for word in words]


def compute_y_tag(sr_tags, tag):
    return sr_tags.apply(lambda x: tag in x)


# MODELS
def vectoriser_to_dict(x_train, x_test, vectorizer):
    return {'vectorizer': vectorizer,
            'x_train': vectorizer.transform(x_train),
            'x_test': vectorizer.transform(x_test),
            'vocabulary': vectorizer.get_feature_names_out()
            }


def get_classifier(kernel: str, kernel_params=None):
    """Retourne le classifieur correspondant au kernel demandé

    Args:
        kernel (str): string correspondant au classifier désiré
        kernel_params (dict, optional): parameters for the kernel

    Returns:
        _type_: _description_
    """
    if kernel_params is None:
        kernel_params = {}
    kernel = kernel.lower().replace(' ', '')
    if kernel == 'dummy':
        clf = DummyClassifier(strategy='most_frequent', **kernel_params)
    elif kernel == 'svc':
        clf = SVC(kernel="linear", **kernel_params)
    elif kernel == 'logisticregression':
        clf = LogisticRegression(solver='sag', **kernel_params)
    elif kernel == 'sdgclassifier':
        clf = SGDClassifier(**kernel_params)
    elif kernel == 'multinomialnb':
        clf = MultinomialNB(**kernel_params)
    elif kernel == 'perceptron':
        clf = Perceptron(**kernel_params)
    elif kernel == 'passiveaggressiveclassifier':
        clf = PassiveAggressiveClassifier(**kernel_params)
    else:
        print('ERROR: unknown kernel:', kernel)
        return
    return clf


class Vectorised():
    """Class pour vectoriser les données texte et gérer la modélisation
    """
    def __init__(self):
        self.x = {}
        self.scaler = None
        self.b_scale_x = False
        self.reset_model()
        self.reset_pca()

    def initialize_data(self, method: str, x_train: Series, y_train: Series,
                        **dict_kwargs: dict):
        b_to_fit = True
        method = method.lower()
        if method == 'countvectorizer':
            vectorizer = CountVectorizer
        elif method == 'tfidf':
            vectorizer = TfidfVectorizer
        elif method == 'word2vec':
            b_to_fit = False
            vectorizer = KeyedVectors.load(
                            'data/models/Word2Vect.model')
        elif method == 'sbert':
            b_to_fit = False
            vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
        elif method == 'use':
            b_to_fit = False
            module_url = "https://tfhub.dev/google/"\
                         "universal-sentence-encoder/4"
            vectorizer = hub.load(module_url)
        else:
            print('VECTORISER ERROR: method should be "countvectoriser" or'
                  '"tfidf"')
            return
        if b_to_fit:
            if dict_kwargs is None:
                dict_kwargs = {}
            self.vectorizer = vectorizer(**dict_kwargs)
            self.vectorizer.fit(x_train)
            print(f'vocaublary length: {len(self.get_vocabulary())}')
        else:
            self.vectorizer = vectorizer
        self.method = method
        self.x = {}
        self.set('train', x_train)
        self.y_train = y_train.copy()
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(self.get('train'))
        self.reset_model()
        self.reset_pca()

    def copy_data_from_vetorizer(self, vectorized):
        """copy data from another vectorized. This method is used to update the
        class without re-train the vectorizer and the classifier.

        Args:
            vectorized (Vectorized): _description_
        """
        # for attribut, value in vectorized.__dict__.items():
        #     setattr(self, attribut, value)
        self.copy_data_from_dict(vectorized.__dict__)

    def copy_data_from_dict(self, dict):
        """copy data from another vectorized. This method is used to update the
        class without re-train the vectorizer and the classifier.

        Args:
            dict (dict): _description_
        """
        for attribut, value in dict.items():
            setattr(self, attribut, value)

    def reset_pca(self):
        self.pca = None
        self.use_pca = False

    def reset_model(self):
        self.clf = None
        self.optimized_thresholds = None

    def get_vocabulary(self):
        return self.vectorizer.get_feature_names_out()

    def transform(self, x):
        if self.method == 'word2vec':
            out = np.empty((len(x), 300), dtype=float)
            for out_i, xi in zip(out, x):
                out_i[:] = self.vectorizer.get_mean_vector(xi.split(' '))
            return out
        elif self.method == 'sbert':
            # AJOUT DE BOUCLE FOR POUR UN SUIVI
            x = x.values
            v0 = self.vectorizer.encode(x[:1])
            out = np.empty_like(v0, shape=(len(x), len(v0[0])))
            out[0, :] = v0[:]
            print('SBERT vectorization:')
            for i in range(1, len(x), 10):
                print(f'{i/len(x):.2%}', end='\r')
                out[i:i+10] = self.vectorizer.encode(x[i:i+10])
            print(f'{1:.2%}    ', end='\n'*2)
            # return self.vectorizer.encode(x.values)
            return out
        elif self.method == 'use':
            # return self.vectorizer(x.values)
            # AJOUT DE BOUCLE FOR POUR UN SUIVI
            x = x.values
            v0 = self.vectorizer(x[:1])
            out = np.empty_like(v0, shape=(len(x), len(v0[0])))
            out[0, :] = v0[:]
            print('USE vectorization:')
            for i in range(1, len(x), 10):
                print(f'{i/len(x):.2%}', end='\r')
                out[i:i+10] = self.vectorizer(x[i:i+10])
            print(f'{1:.2%}    ', end='\n'*2)
            # return self.vectorizer.encode(x.values)
            return out
        else:
            return self.vectorizer.transform(x)

    def set(self, which: str, x):
        self.x[which] = self.transform(x)

    def get(self, which: str):
        x = self.x[which]
        if hasattr(x, 'todense'):
            x = x.todense()
        return np.asarray(x)

    def scale(self, x):
        if self.b_scale_x:
            if self.use_pca:
                return self.pca.transform(self.scaler.transform(x))
            return self.scaler.transform(x)
        return x

    def pca_fit_on_train(self, n_components: int):
        use_pca = self.use_pca
        self.use_pca = False
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.scale(self.get('train')))
        self.use_pca = use_pca

    def train_classifier(self, kernel: str, b_scale_x: bool = True,
                         n_pca_components=0, kernel_params=None):
        clf = get_classifier(kernel, kernel_params)
        if clf is None:
            return
        self.use_pca = n_pca_components > 0
        if self.use_pca:
            self.pca_fit_on_train(n_pca_components)
        clf = OneVsRestClassifier(clf, n_jobs=-1)
        self.b_scale_x = b_scale_x
        clf.fit(self.scale(self.get('train')), self.y_train)
        self.clf = clf

    def grid_search_cv(self, kernel: str, param_grid: dict,
                       b_scale_x: bool = True):
        clf = get_classifier(kernel)
        clf = OneVsRestClassifier(clf)
        self.b_scale_x = b_scale_x
        clf = model_selection.GridSearchCV(clf, param_grid,
                                           cv=5,
                                           n_jobs=-1,
                                           scoring=metrics.f1_score,
                                           )
        clf.fit(self.scale(self.get('train')), self.y_train)
        self.clf = clf.best_estimator
        print('best parameters:')
        print(clf.best_params_)

    def optimize_threshold(self):
        """Compute optimized threshold value (on F1_score) and store the values

        Returns:
            vector: f1_score for each column of y used to train the model
        """
        y_pred_proba = self.predict_proba('train')
        threshold_score = compute_best_threshold_and_score_per_output(
                                                        self.y_train.values,
                                                        y_pred_proba)
        threshold_score[:, 1] *= 100.
        self.optimized_thresholds = threshold_score[:, 0]
        return threshold_score[:, 1]

    def predict(self, which: str):
        return self.clf.predict(self.scale(self.get(which)))

    def predict_proba(self, which: str):
        return self.clf.predict_proba(self.scale(self.get(which)))

    def predict_optim_threshold(self, which: str):
        if self.optimized_thresholds is None:
            self.optimize_threshold()
        y_pred_proba = self.predict_proba(which)
        y_pred = np.empty_like(y_pred_proba, dtype=int)
        for y_pred_i, y_pred_proba_i, threshold \
                in zip(y_pred.T,
                       y_pred_proba.T,
                       self.optimized_thresholds):
            y_pred_i[:] = y_pred_proba_i > threshold
        return y_pred

    def get_prediction_tags(self):
        return self.y_train.columns


def train_extra_trees(vectorised: Vectorised, param_grid: dict):
    """Effectue un GridSearchCV avec un modèle ExtraTrees en fonction
    des paramètres fournis

    Args:
        vectorised (Vectorised): données d'entrée
        param_grid (dict): paramètres pour le grisearch

    Returns:
        classifier: meilleur modèle
    """
    x_train = vectorised.scale(vectorised.get('train'))
    y_train = vectorised.y_train
    print('x_train:', x_train.shape)
    print('y_train:', y_train.shape)
    trees = ExtraTreesClassifier(n_jobs=-1, n_estimators=300)
    trees = model_selection.GridSearchCV(
                trees,
                param_grid,
                cv=5,
                n_jobs=1,
                scoring='f1_macro',
    )
    # x_train = x_train[:10000]
    # y_train = y_train[:x_train.shape[0]]
    print('x_train:', x_train.shape)
    print('y_train:', y_train.shape)
    trees.fit(x_train, y_train)
    print('best parameters:')
    print(trees.best_params_)
    return trees.best_estimator_


def plot_f1_scores(results: dict,
                   tag_names: list,
                   title: str,
                   savename: str):
    """Affiche et sauvegarde un graphique pour comparer les F1 scores pour
    différentes configurations

    Args:
        results (dict): différentes résultats
        tag_names (list): nom des classes / tags
        title (str): titre du graphique (scores F1 est ajouté avant)
        savename (str): nom de fichier de sauvegarde (sans extension,
                        un prefix est ajouté)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel('tags')
    ax.set_ylabel('F1 scores (%)')
    for model, values in results.items():
        ax.plot(tag_names, values, '-o', label=model)
    ax.legend()
    ax.set_title(f'scores F1 {title}')
    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'F1_scores_'
                        + savename + '.pdf')
    return


def plot_ebouli_pca(pca, inertie_target: float, savename: str):
    """Trace et sauvegade un ébouli d'intertie pour une pca fournie

    Args:
        pca (PCA sklearn): PCA entrainée
        inertie_target (float): valeur cible (%) d'inertie cumulée
        savename (str): nom de fichier pour la sauvegarde
    """
    scree = (pca.explained_variance_ratio_*100)
    inertie = scree.cumsum()
    x_list = range(1, scree.size+1)
    # print('scree:', scree.round(2))
    # print('sum scree:', scree.sum().round(2))
    nmax = (inertie < inertie_target).sum() + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x_list, scree)
    ax.plot(x_list, inertie, c='b',  # marker='o',
            label='intertie cumulée')
    ax.vlines(nmax, inertie.min(), inertie.max(), color='y', linestyle='--')
    # ax.hlines(inertie_target, 0.9*nmax, 1.1*nmax, color='y', linestyle='--')
    ax.annotate(f"{inertie_target}% d'inertie\npour\n{nmax} composantes",
                [nmax*0.95, inertie_target*0.5],
                ha='right', va='center', color='y')
    ax.legend()
    ax.set_xlabel("rang de l'axe d'inertie")
    ax.set_ylabel("inertie (%)")
    ax.set_title('Éboulis des valeurs propres')
    fig.tight_layout(pad=0.2)
    graph_tools.savefig(fig, PATH_PRINT + savename + '.pdf')


def count_vectorizer(x_train, x_test, max_features):
    cvect = CountVectorizer(max_df=1., min_df=100,
                            max_features=max_features)
    cvect = cvect.fit(x_train)
    return vectoriser_to_dict(x_train, x_test, cvect)


def tfidf_vectorizer(x_train, x_test, max_features):
    ctf = TfidfVectorizer(max_df=1., min_df=100,
                          max_features=max_features)
    ctf = ctf.fit(x_train)
    return vectoriser_to_dict(x_train, x_test, ctf)


def init_lda(vectorised, n_topics):
    """Compute LDA and init visualisation from vectorised
    data for n_topics

    Args:
        vectorised (dict): output of vectoriser_to_dict func.
        n_topics (int): number of topics

    Returns:
        LDAvis: LDAvis object to be displayed
    """
    print(f'LDA: {n_topics} topics')
    x = vectorised.x['train']
    vectorizer = vectorised.vectorizer
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                    learning_method='online', verbose=True)
    data_lda = lda.fit_transform(x)

    vis = pyLDAvis.lda_model.prepare(lda, x, vectorizer, mds='tsne')
    return vis


def save_LDAvis(vis):
    with open(PATH_PRINT + 'LDAvis.html', 'w') as file:
        pyLDAvis.save_html(vis, file)


def show_lda_vis(vis):
    return pyLDAvis.display(vis)


def compute_simple_metric(y_true, y_pred) -> tuple:
    """compute precision, recall and scpecificity scores

    Args:
        y_true (array): true labels
        y_pred (array): predicted labels

    Returns:
        tuple: precision, recall, specificity, F1 score
    """
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    # print('TN:', TN, 'FP:', FP, 'FN:', FN, 'TP:', TP)
    recall = 100 * TP / max(1, TP+FN)
    precision = 100 * TP / max(1, TP+FP)
    specificity = 100 * TN / max(1, TN + FP)
    F1score = 100*(TP+TP) / max(1, TP+TP+FP+FN)
    return precision, recall, specificity, F1score


def simple_metric_dataframe(y_true, y_pred, prediction_tags) -> DataFrame:
    """compute simple metric and return results in a DataFrame

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        prediction_tags (_type_): _description_

    Returns:
        DataFrame: _description_
    """
    results = pd.DataFrame(index=['precision', 'recall', 'specificity',
                                  'F1 score'])
    for y_true_i, y_pred_i, tag in zip(y_true.T, y_pred.T, prediction_tags):
        # print(funcs.compute_simple_metric(y_true[:,i], y_pred[:,i]))
        results[tag] = compute_simple_metric(y_true_i, y_pred_i)
    return results.round(2)


def compute_best_threshold_and_score(y_true, y_pred_proba, metric_func=None):
    """compute best threshold value to optimise a given score

    Args:
        y_true (_type_): _description_
        y_pred_proba (_type_): _description_
        metric_func (func): function returning the desired metric.
                            Default is f1_score

    Returns:
        _type_: _description_
    """
    if metric_func is None:
        metric_func = metrics.f1_score

    score, threshold = 0, 0.
    for i in range(1, 100):
        threshold_i = i * 0.01
        score_i = metric_func(y_true, (y_pred_proba > threshold_i).astype(int))
        if score_i > score:
            score, threshold = score_i, threshold_i
    return threshold, score


def compute_best_threshold_and_score_per_output(y_true, y_pred_proba,
                                                metric_func=None):
    """compute best threshold and score for each output of input data

    Args:
        y_true (array): (n_sample x n_outputs)
        y_pred_proba (array): probability outputs of a model
        metric_func (cuntion, optional): function which return the desired
                                         metric. Defaults to None -> f1_score.

    Returns:
        numpy array: (n_sample x 2), [theshold, score]
    """
    if metric_func is None:
        metric_func = metrics.f1_score
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred_proba = y_pred_proba.reshape(-1, 1)

    out = np.empty((y_true.shape[1], 2), dtype=float)
    for out_i, y_true_i, y_pred_proba_i in zip(out,
                                               y_true.transpose(),
                                               y_pred_proba.transpose()):
        out_i[:] = compute_best_threshold_and_score(y_true_i, y_pred_proba_i)
    return out


def multi_label_binarize(tags, selected_tags):
    """generate a DataFrame len(docs) x len(selected tags)
    each column corresponds to a selected tag
    each row corresponds to an element

    Args:
        tags (array of list): array of list of tags for each element
        selected_tags (list): list of selected tags (classes) for the
                              MultiLabelBinarizer

    Returns:
        DataFrame: Multi Label Binarized data
    """
    binarizer = preprocessing.MultiLabelBinarizer(classes=selected_tags)
    x = binarizer.fit_transform(tags)
    return pd.DataFrame(x, columns=selected_tags)


def plot_sum_vectorizer(x, vectorised):
    x0 = x.values
    x = vectorised.get('train')
    vocabulary = vectorised.get_vocabulary()
    print('x:', x.shape)
    print('vocabulary:', len(vocabulary))

    x_sum_0 = np.asarray(x.sum(axis=0)).ravel()
    print('\nx sum axis=0:', x_sum_0.shape)
    inds_sort = x_sum_0.argsort()
    x_sum_0.sort()

    vmax = 0.05 * x_sum_0[-1]
    inds_plot = np.arange(x_sum_0.size)[x_sum_0 > vmax]

    print('most used terms (decreasing order)')
    print(np.asarray(vocabulary)[inds_sort[inds_plot][::-1]])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x_sum_0)
    # ax.plot(inds_plot, x_sum_0.T[inds_plot], 'r')
    ax.set_title('Fréquence de chaque terme')
    ax.set_ylabel("nombre d'occurence")

    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'freq_termes.pdf')

    x_sum_1 = np.asarray(x.sum(axis=1)).ravel()
    b_is_nul_sum_1 = x_sum_1 == 0
    print('\nx sum axis=1:', x_sum_1.shape)
    print('number of null:', b_is_nul_sum_1.sum())
    x_sum_1.sort(0)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x_sum_1)
    ax.set_title("Nombre d'élements du vocabulaire dans chaque document")
    ax.set_ylabel("nombre d'éléments")

    fig.tight_layout()
    graph_tools.savefig(fig, PATH_PRINT + 'freq_vocab_doc.pdf')

    # éléments nulls
    print('\nEntrées sans élement du vocabulaire:')
    for i in np.arange(x0.size)[b_is_nul_sum_1]:
        print(x0[i], '\n')


# LatentDirichletAllocation(n_components=10, *, doc_topic_prior=None,
# topic_word_prior=None, learning_method='batch', learning_decay=0.7,
# learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1,
# total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001,
# max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None)
# def train_lda(x, n_tags):
#     decomposition.LatentDirichletAllocation
#     lda = decomposition.LatentDirichletAllocation(n_components=n_tags,
#                                                   random_state=0)
#     lda.fit(x)
#     return lda
#  |  >>> # get topics for some given samples:
#  |  >>> lda.transform(X[-2:])
#  |  array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
#  |         [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])


# %% MULTILABELS

# inspiré de l'example "multilabel classification"
# Authors: Vlad Niculae, Mathieu Blondel
# License: BSD 3 clause
def plot_hyperplane(ax, clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    ax.plot(xx, yy, linestyle, label=label)


# def plot_subfigure_multilabel


def transform_and_train_classifier(transform=None):
    """Initialisation d'un pipieline avec ou sans transformation initiale

    Args:
        transform (str, optional): 'pca', 'cca' ou None . Defaults to None.

    Returns:
        pipeline: modèle avec ou sans transfomer initial
    """
    process = [('OneVsRestClassifier',
                OneVsRestClassifier(SVC(kernel="linear"))
                )]
    if transform is not None:
        if transform.lower() == "pca":
            process.insert(0, ('pca', PCA(n_components=2)))
        elif transform.lower() == "cca":
            process.insert(0, ('cca', CCA(n_components=2)))
        elif transform.lower() == 'svd':
            process.insert(0, ('svd', TruncatedSVD(n_components=2)))
    pipe = Pipeline(process)
    return pipe

def x_vectorized_to_array(x):
    return np.asarray(x.todense())

def vectorized_add_scaler(vectorised):
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_vectorized_to_array(vectorised['x_train']))
    vectorised['scaler'] = scaler

def vectorized_get_x_scaled(vectorized: dict, which: str):
    x = x_vectorized_to_array(vectorized[f'x_{which}'])
    return vectorised['scaler'].transform(x)

def pca_on_vectorized(vectorized, n_components):
    x = vectorized_get_x_scaled(vectorized, 'train')
    pca = PCA(n_components=n_components)
    pca.fit(x)

# TODO: docstring
def process_mutlilabel_classifier(vectorized, y_train, y_test, transform=None):
    x_train = np.asarray(vectorized['x_train'].todense())
    clf = transform_and_train_classifier(transform)
    clf.fit(x_train, y_train)
    L_train = clf.predict(x_train)
    L_test = clf.predict(vectorized['x_test'].todense())
    return {'clf': clf, 'L_train': L_train, 'L_test': L_test}
    # ARI = metrics.adjusted_rand_score(y_train, L_train)
    # print(f'train ARI: {ARI:.4f}')
    # ARI = np.round(metrics.adjusted_rand_score(y_test, L_test), 4)
    # print(f'test ARI: {ARI:.4f}')


def score_multilabel_classifier(data, y_train, y_test):
    L_train = data['L_train']
    L_test = data['L_test']




# Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie
# et n° de clusters
def ARI_fct(features, l_cat, y_cat_num):
    time1 = time.time()
    num_labels = len(l_cat)
    tsne = manifold.TSNE(n_components=2, perplexity=30, n_iter=2000,
                         init='random', learning_rate=200,
                         random_state=42)
    X_tsne = tsne.fit_transform(features)

    # Détermination des clusters à partir des données après Tsne
    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)
    ARI = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_), 4)
    time2 = np.round(time.time() - time1, 0)
    print("ARI : ", ARI, "time : ", time2)

    return ARI, X_tsne, cls.labels_


# visualisation du Tsne selon les vraies catégories et selon les clusters
def TSNE_visu_fct(X_tsne, y_cat_num, labels, ARI):
    fig = plt.figure(figsize=(15, 6))

    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=l_cat, loc="best",
              title="Categorie")
    plt.title('Représentation des tweets par catégories réelles')

    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels),
              loc="best", title="Clusters")
    plt.title('Représentation des tweets par clusters')

    plt.show()
    print("ARI : ", ARI)

# %% END OF FILE
###
