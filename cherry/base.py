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
from .exceptions import StopWordsNotFoundError, UnicodeFileEncodeError, CacheNotFoundError

CHERRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cherry')
DATA_DIR = os.path.join(CHERRY_DIR, 'data')

def stop_words(filename):
    '''
    Return Stop words depent on filename
    stop_word('chinese_classify.dat') will return the data in
    DATA_DIR/stop_words_chinese_classify.dat
    '''
    try:
        stop_words_path = os.path.join(DATA_DIR, filename)
        with open(stop_words_path, encoding='utf-8') as f:
            stop_words = [l[:-1] for l in f.readlines()]
    except IOError:
        error = 'Stop words file not found'
        raise StopWordsNotFoundError(error)
    except UnicodeEncodeError as e:
        error = e
        raise UnicodeFileEncodeError(error)
    return stop_words

def load_data(filename):
    '''
    TODO: use a generator instead
    '''
    text, label = [], []
    with open(os.path.join(DATA_DIR, filename)) as file:
        for line in file.readlines():
            row = line.split('\n')[0].rsplit(',', 1)
            text.append(row[0])
            label.append(row[1])
    return np.asarray(text), np.asarray(label)

def write_file(self, path, data):
    '''
    Write data to path
    '''
    with open(path, 'w') as f:
        f.write(data)

def load_cache(filename):
    '''
    Load cache data from file
    '''
    cache_path = os.path.join(DATA_DIR, filename)
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        error = (
            'Cache files not found,' +
            'maybe you should train the data first.')
        raise CacheNotFoundError(error)
