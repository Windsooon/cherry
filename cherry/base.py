# -*- coding: utf-8 -*-

"""
cherry.base
~~~~~~~~~~~~
Base method for cherry classify
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""
import os
import numpy as np
import pickle
from .exceptions import FilesNotFoundError, UnicodeFileEncodeError, \
    CacheNotFoundError, MethodNotFoundError, DataMismatchError

CHERRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cherry')
DATA_DIR = os.path.join(CHERRY_DIR, 'data')

def get_stop_words(model):
    '''
    Return Stop words depent on filename
    stop_word('chinese_classify.dat') will return the data in
    DATA_DIR/stop_words_chinese_classify.dat
    '''
    model_path = os.path.join(DATA_DIR, model)
    for stop_words_file in os.listdir(model_path):
        if stop_words_file.startswith('stop_words'):
            try:
                with open(os.path.join(model_path, stop_words_file), encoding='utf-8') as file:
                    stop_words = [l[:-1] for l in file.readlines()]
            except UnicodeEncodeError as e:
                error = e
                raise UnicodeFileEncodeError(error)
            return stop_words
    error = 'Stop words file not found'
    raise FilesNotFoundError(error)

def load_data(model):
    '''
    TODO: use a generator instead
    '''
    text, label = [], []
    model_path = os.path.join(DATA_DIR, model)
    for data_file in os.listdir(model_path):
        if data_file.startswith('data'):
            with open(os.path.join(model_path, data_file)) as file:
                for line in file.readlines():
                    row = line.split('\n')[0].rsplit(',', 1)
                    if len(row) != 2:
                        error = 'data mismatch in {0}, make sure there is a "," before the category index'.format(row)
                        raise DataMismatchError(error)
                    text.append(row[0])
                    label.append(row[1])
            return np.asarray(text), np.asarray(label)
    error = 'Data file not found'
    raise FilesNotFoundError(error)

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
