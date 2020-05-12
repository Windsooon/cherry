# -*- coding: utf-8 -*-

"""
cherry.base
~~~~~~~~~~~~
Base method for cherry classify
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""
import os
import pickle
import numpy as np

from urllib.request import urlretrieve
from .exceptions import FilesNotFoundError, UnicodeFileEncodeError, \
    CacheNotFoundError, MethodNotFoundError, DataMismatchError
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.datasets.base import load_files

CHERRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cherry')
DATA_DIR = os.path.join(CHERRY_DIR, 'dataset')
BUILD_IN_MODELS = {'4sen-harmful': ('abc.com', 'harmful')}

__all__ = ['get_stop_words',
           'load_data',
           'write_file',
           'load_cache',
           'tokenizer',
           'get_vectorizer',
           'get_clf']

def get_stop_words(language=None):
    '''
    TODO: add IDF after every stop word
    '''
    if not language:
        return ENGLISH_STOP_WORDS
    else:
        return stop_words[language]

def load_data(model):
    model_data_path = os.path.join(DATA_DIR, model)
    if os.path.exists(model_data_path):
        data = _load_data_from_local(model)
    elif model in BUILD_IN_MODELS:
        data = _load_data_from_remote(model)
    else:
        error = '{0} is not built in models and not found in dataset folder.'.format(model)
        raise FilesNotFoundError(error)
    return data

def _load_data_from_local(model):
    info = BUILD_IN_MODELS[model]
    bunch = load_files(path)
    return bunch

def _load_data_from_remote(mode):
    info = BUILD_IN_MODELS[model]
    _download_data(info.url, info.path)
    return _load_data_from_local(model)

def _download_data(url, path, times):
    count = 0
    while 1:
        try:
            path = urlretrieve(url, path)
        except:
            count += 1
            if count == times:
                raise
    return path

def write_file(self, path, data):
    '''
    Write data to path
    '''
    with open(path, 'w') as f:
        f.write(data)

def load_cache(model, type):
    '''
    Load cache data from file
    '''
    if type == 'trained.pkl':
        cache_path = os.path.join(DATA_DIR, model + '/trained.pkl')
    else:
        cache_path = os.path.join(DATA_DIR, model + '/ve.pkl')
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        error = (
            'Cache files not found,' +
            'maybe you should train the data first.')
        raise CacheNotFoundError(error)

def tokenizer(text):
    '''
    You can use your own tokenizer function here, by default,
    this function only work for chinese
    '''
    # For English, try:
    # from nltk.tokenize import word_tokenize
    # return [t.lower() for t in word_tokenize(text) if len(t) > 1]
    import jieba
    return [t for t in jieba.cut(text) if len(t) > 1]

def get_vectorizer(model, vectorizer_method):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
    if not vectorizer_method:
        vectorizer_method = 'Count'
    mapping = {
        'Count': CountVectorizer,
        'Tfidf': TfidfVectorizer,
        'Hashing': HashingVectorizer,
    }
    try:
        method = mapping[vectorizer_method]
    except KeyError:
        raise MethodNotFoundError
    else:
        return method(tokenizer=tokenizer, stop_words=get_stop_words(model))

def get_clf(model, clf_method):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    if not clf_method:
        clf_method = 'MNB'
    mapping = {
        'MNB': (MultinomialNB, {'alpha': 0.1}),
        'SGD': (SGDClassifier, {'loss': 'hinge', 'penalty': 'l2', 'alpha': 1e-3, 'max_iter': 5, 'tol': None}),
        'RandomForest': (RandomForestClassifier, {'max_depth': 5}),
        'AdaBoost': (AdaBoostClassifier, {}),
    }
    try:
        method, parameters = mapping[clf_method]
    except KeyError:
        raise MethodNotFoundError
    else:
        return method(**parameters)
