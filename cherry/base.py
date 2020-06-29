# -*- coding: utf-8 -*-

"""
cherry.base
~~~~~~~~~~~~
Base method for cherry
:copyright: (c) 2018-2020 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""
import os
import pickle
import tarfile
import hashlib
import codecs
import urllib
import logging
import numpy as np

from collections import namedtuple
from urllib.request import urlretrieve
from .exceptions import *
from .common import *
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import load_files

CHERRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cherry')
DATA_DIR = os.path.join(os.getcwd(), 'datasets')

__all__ = ['DATA_DIR',
           'get_stop_words',
           'load_data',
           'write_file',
           'load_all',
           'load_cache',
           'get_vectorizer_and_clf',
           'get_tokenizer',
           'get_vectorizer',
           'get_clf']

def get_stop_words(language='English'):
    '''
    There are several known issues in our provided ‘english’ stop word list.
    It does not aim to be a general, ‘one-size-fits-all’ solution as some
    tasks may require a more custom solution.
    See https://aclweb.org/anthology/W18-2502 for more details.
    TODO: add IDF after every stop word.
    '''
    if language == 'English':
        return ENGLISH_STOP_WORDS
    try:
        return STOP_WORDS[language]
    except KeyError:
        error = 'Cherry didn\'t support {0} at this moment.'.format(language)
        raise NotSupportError(error)

def load_all(model, language=None, preprocessing=None, categories=None, encoding=None, vectorizer=None,
            vectorizer_method=None, clf=None, clf_method=None, x_data=None, y_data=None):
    # If user didn't pass x_data and y_data, try to load data from local or remote
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not (x_data and y_data):
        try:
            cache = load_data(model, categories=categories, encoding=encoding)
        except FilesNotFoundError:
            error = ('Please make sure your put the {0} data inside `datasets` '
                    'folder or use model inside "email", "review" or "newsgroups".'.format(model))
            raise FilesNotFoundError(error)
        if preprocessing:
            cache.data = [preprocessing(text) for text in cache.data]
        x_data, y_data = cache.data, cache.target
    vectorizer, clf = get_vectorizer_and_clf(
        language, vectorizer, clf,
        vectorizer_method, clf_method)
    return x_data, y_data, vectorizer, clf

def load_data(model, categories=None, encoding=None):
    '''
    Load data using `model` name
    '''
    model_path = os.path.join(DATA_DIR, model)
    if os.path.exists(model_path):
        return _load_data_from_local(
            model, categories=categories, encoding=encoding)
    else:
        return _load_data_from_remote(
            model, categories=categories, encoding=encoding)

def _load_data_from_local(
        model, categories=None, encoding=None):
    '''
    1. Find local cache files
    2. If we can't find the cache files
           3.1 Try to create cache files using data files inside `datasets`.
           2.2 Raise error if create cache files failed.
    '''
    model_path = os.path.join(DATA_DIR, model)
    cache_path = os.path.join(model_path, model + '.pkz')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                compressed_content, 'zlib_codec')
            return pickle.loads(uncompressed_content)['all']
        except Exception as e:
            # Can't load cache files
            error = ('Can\'t load cached data from {0}. '
                    'Please try again after delete cache files.'.format(model))
            raise NotSupportError(error)
    cache = dict(all=load_files(
        model_path, categories=categories, encoding=encoding))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)
    return cache['all']

def _load_data_from_remote(model, categories=None, encoding=None):
    try:
        info = BUILD_IN_MODELS[model]
    except KeyError:
        error = ('{0} is not in BUILD_IN_MODELS.').format(model)
        raise FilesNotFoundError(error)
    # The original data can be found at:
    # https://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz
    meta_data_c = namedtuple('meta_data_c', ['filename', 'url', 'checksum', 'encoding'])
    # Create a nametuple
    meta_data = meta_data_c(filename=info[0], url=info[1], checksum=info[2], encoding=info[3])
    _fetch_remote(meta_data, DATA_DIR)
    _decompress_data(meta_data.filename, DATA_DIR)
    return _load_data_from_local(
        model, categories=categories, encoding=info[3])

def _fetch_remote(remote, dirname=None):
    """
    Function from sklearn
    Helper function to download a remote datasets into path
    Copy from sklearn.datasets.base
    """

    file_path = (remote.filename if dirname is None
                 else os.path.join(dirname, remote.filename))
    print('Downloading data from {0}'.format(remote.url))
    urlretrieve(remote.url, file_path)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote.checksum))
    return file_path

def _sha256(path):
    """
    Function from sklearn
    Calculate the sha256 hash of the file at path.
    """
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()

def _decompress_data(filename, model_path):
    '''
    Function from sklearn
    '''
    file_path = os.path.join(model_path, filename)
    logging.debug("Decompressing %s", file_path)
    tarfile.open(file_path, "r:gz").extractall(path=model_path)
    os.remove(file_path)

def _train_test_split(cache, test_size=0.1):
    data_lst = list()
    target = list()
    filenames = list()
    data = cache['all']
    data_lst.extend(data.data)
    target.extend(data.target)
    filenames.extend(data.filenames)
    data.data = data_lst
    data.target = np.array(target)
    data.filenames = np.array(filenames)
    return train_test_split(data.data, data.target, test_size=test_size, random_state=0)

def write_file(path, data):
    '''
    Write data to path
    '''
    with open(path, 'a+') as f:
        f.write(data)

def write_cache(model, content, path):
    '''
    Write cached file under model dir
    '''
    cache_path = os.path.join(DATA_DIR, model + '/' + path)
    compressed_content = codecs.encode(pickle.dumps(content), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

def load_cache(model, path):
    '''
    Load cache data from file
    '''
    cache_path = os.path.join(DATA_DIR, model + '/' + path)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                compressed_content, 'zlib_codec')
            return pickle.loads(uncompressed_content)
        except Exception as e:
            error = (
                'Can\'t load cached files.')
            raise CacheNotFoundError(error)
    else:
        error = (
            'Can\'t find cache files')
        raise CacheNotFoundError(error)

def english_tokenizer_wrapper(text):
    from nltk.tokenize import word_tokenize
    return [t for t in word_tokenize(text) if len(t) > 1]

def chinese_tokenizer_wrapper(text):
    import jieba
    return [t for t in jieba.cut(text) if len(t) > 1]

def get_tokenizer(language):
    if language == 'English':
        return english_tokenizer_wrapper
    elif language == 'Chinese':
        return chinese_tokenizer_wrapper
    else:
        raise NotSupportError((
            'You need to specify tokenizer function ' +
            'when the language is nor English or Chinese.'))

def get_vectorizer_and_clf(
    language, vectorizer, clf, vectorizer_method, clf_method):
    if not vectorizer:
        vectorizer = get_vectorizer(language, vectorizer_method)
    if not clf:
        clf = get_clf(clf_method)
    return vectorizer, clf

def get_vectorizer(language, vectorizer_method):
    mapping = {
        'Count': CountVectorizer,
        'Tfidf': TfidfVectorizer,
        'Hashing': HashingVectorizer,
    }
    try:
        method = mapping[vectorizer_method]
    except KeyError:
        error = 'Please make sure vectorizer_method in "Count", "Tfidf" or "Hashing".'
        raise MethodNotFoundError(error)
    else:
        return method(tokenizer=get_tokenizer(language), stop_words=get_stop_words(language))

def get_clf(clf_method):
    mapping = {
        'MNB': (MultinomialNB, {'alpha': 0.1}),
        'SGD': (SGDClassifier, {'loss': 'hinge', 'penalty': 'l2', 'alpha': 1e-3, 'max_iter': 5, 'tol': None}),
        'RandomForest': (RandomForestClassifier, {'max_depth': 5}),
        'AdaBoost': (AdaBoostClassifier, {}),
    }
    try:
        method, parameters = mapping[clf_method]
    except KeyError:
        error = 'Please make sure clf_method in "MNB", "SGD", "RandomForest" or "AdaBoost".'
        raise MethodNotFoundError(error)
    return method(**parameters)
