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
import urllib
import numpy as np

from urllib.request import urlretrieve
from .exceptions import *
from cherry.datasets import STOP_WORDS, BUILD_IN_MODELS
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets.base import load_files

CHERRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cherry')
DATA_DIR = os.path.join(CHERRY_DIR, 'datasets')

__all__ = ['DATA_DIR',
           'get_stop_words',
           'load_data',
           'write_file',
           'load_cache',
           '_load_data_from_local',
           '_load_data_from_remote',
           '_download_data',
           'tokenizer',
           'get_vectorizer',
           'get_clf']

def get_stop_words(language='English'):
    '''
    There are several known issues in our provided ‘english’ stop word list. It does not aim to be a general, ‘one-size-fits-all’ solution as some tasks may require a more custom solution. See https://aclweb.org/anthology/W18-2502 for more details.
    TODO: add IDF after every stop word.
    '''
    if language == 'English':
        return ENGLISH_STOP_WORDS
    try:
        stop_word = STOP_WORDS[language]
    except KeyError:
        error = 'Cherry didn\'t support {0} at this moment.'.format(language)
        raise NotSupportError(error)

def load_data(model, categories=None, encoding=None):
    '''
    Load data using `model` name
    '''
    path = os.path.join(DATA_DIR, model)
    if os.path.exists(path):
        return _load_data_from_local(path, model, categories=categories, encoding=encoding)
    else:
        return _load_data_from_remote(model, categories=categories, encoding=encoding)

def _load_data_from_local(path, model, categories=None, encoding=None):
    '''
    1. Try to find local cache files
    2. If we can't find the cache files
           3.1 Try to create cache files using data files inside `dataset`.
           2.2 Raise error if create cache files failed.
    '''
    cache_path = os.path.join(path, model + 'pkz')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                compressed_content, 'zlib_codec')
            return pickle.loads(uncompressed_content)
        except Exception as e:
            # Can't load cache files
            pass
    return load_files(path, categories=categories, encoding=encoding)

def _load_data_from_remote(model, categories=None, encoding=None):
    try:
        info = BUILD_IN_MODELS[model]
    except KeyError:
        error = '{0} is not built in models and not found in dataset folder.'.format(model)
        raise FilesNotFoundError(error)
    else:
        _download_data(info[0], info[1], categories, encoding)
    return _load_data_from_local(info[0], model)

def _download_data(url, path, categories, encoding):
    print("Trying to download {1} data files from {0}.".format(url, path))
    try:
        urlretrieve(url, path)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        error = 'Can\' download model form {0}'.format(url)
        raise DownloadError(error)

def write_file(self, path, data):
    '''
    Write data to path
    '''
    with open(path, 'w') as f:
        f.write(data)

def write_cache(model, vectorizer, clf):
    '''
    Write cache file under DATA_DIR
    '''
    cache_path = os.path.join(DATA_DIR, model + '/trained.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(clf, f)
    cache_path = os.path.join(DATA_DIR, model + '/ve.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_cache(model):
    '''
    Load cache data from file
    '''
    cache = None
    cache_path = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            error = (
                'Cache loading failed.')
            raise CacheNotFoundError(error)

def tokenizer(text, language='English'):
    '''
    You can use your own tokenizer function here, by default,
    this function work for English
    '''
    if language == 'English':
        from nltk.tokenize import word_tokenize
        return [t.lower() for t in word_tokenize(text) if len(t) > 1]
    elif language == 'Chinese':
        import jieba
        return [t for t in jieba.cut(text) if len(t) > 1]
    else:
        raise

def get_vectorizer(model, vectorizer_method):
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
