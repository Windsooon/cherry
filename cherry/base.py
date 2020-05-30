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
from cherry.datasets import STOP_WORDS, BUILD_IN_MODELS
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import load_files

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
           'tokenizer',
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

def load_data(model, preprocessing=None, categories=None, encoding=None, split=False):
    '''
    Load data using `model` name
    '''
    model_path = os.path.join(DATA_DIR, model)
    if os.path.exists(model_path):
        try:
            info = BUILD_IN_MODELS[model]
        except KeyError:
            pass
        else:
            if not encoding:
                encoding = info[3]
        return _load_data_from_local(
            model, preprocessing=preprocessing, categories=categories,
            encoding=encoding, split=split)
    else:
        return _load_data_from_remote(
            model, preprocessing=preprocessing, categories=categories,
            encoding=encoding, split=split)

def _load_data_from_local(model, preprocessing=None, categories=None, encoding=None, split=False):
    '''
    1. Try to find local cache files
    2. If we can't find the cache files
           3.1 Try to create cache files using data files inside `dataset`.
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
                    'Please try again after delete those cache files.'.format(model))
            raise NotSupportError(error)
    cache = dict(all=load_files(
        model_path, categories=categories, encoding=encoding))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)
    if split:
        return _train_test_split(cache)
    return cache['all']

def _load_data_from_remote(model, preprocessing=None, categories=None, encoding=None, split=False):
    try:
        info = BUILD_IN_MODELS[model]
    except KeyError:
        error = ('{0} is not built in models and not found '
                'in dataset folder.').format(model)
        raise FilesNotFoundError(error)
    # The original data can be found at:
    # https://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz
    meta_data_c = namedtuple('meta_data_c', ['filename', 'url', 'checksum', 'encoding'])
    # Create a nametuple
    meta_data = meta_data_c(filename=info[0], url=info[1], checksum=info[2], encoding=info[3])
    model_path = os.path.join(DATA_DIR, model)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    _fetch_remote(meta_data, model_path)
    _decompress_data(meta_data.filename, model_path)
    return _load_data_from_local(
        model, preprocessing=preprocessing,
        categories=categories, encoding=info[3],
        split=split)

def _fetch_remote(remote, dirname=None):
    """Helper function to download a remote dataset into path
       Copy from sklearn.datasets.base
    """

    file_path = (remote.filename if dirname is None
                 else os.path.join(dirname, remote.filename))
    urlretrieve(remote.url, file_path)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote.checksum))
    return file_path

def _train_test_split(cache):
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
    return train_test_split(data.data, data.target, test_size=0.2, random_state=0)

def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
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
    file_path = os.path.join(model_path, filename)
    logging.debug("Decompressing %s", file_path)
    tarfile.open(file_path, "r:gz").extractall(path=model_path)
    os.remove(file_path)

def write_file(self, path, data):
    '''
    Write data to path
    '''
    with open(path, 'w') as f:
        f.write(data)

def write_cache(model, content, path):
    '''
    Write trained file under model DATA_DIR
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
                'Cache loading failed.')
            raise CacheNotFoundError(error)
    else:
        error = (
            'Can\'t find cache files')
        raise CacheNotFoundError(error)

def tokenizer(text, language='English'):
    '''
    English: nltk
    Chinese: jieba
    '''
    if language == 'English':
        from nltk.tokenize import word_tokenize
        return [t.lower() for t in word_tokenize(text) if len(t) > 1]
    elif language == 'Chinese':
        import jieba
        return [t for t in jieba.cut(text) if len(t) > 1]
    else:
        raise

def get_vectorizer(model, language, vectorizer_method):
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
        return method(tokenizer=tokenizer, language=language, stop_words=get_stop_words(language))

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
