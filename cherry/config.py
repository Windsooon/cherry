# -*- coding: utf-8 -*-

"""
cherry.config
~~~~~~~~~~~~
Base config of cherry classify
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import jieba
from .exceptions import StopWordsNotFoundError
from sklearn.feature_extraction.text import CountVectorizer

def _stop_word(dir):
    '''
    Stop word located in the stop_words.dat file
    '''
    try:
        stop_word_path = os.path.join(dir, 'stop_words.dat')
        with open(stop_word_path, encoding='utf-8') as f:
            stop_word = [l[:-1] for l in f.readlines()]
    except IOError:
        error = 'stop_words.dat not found'
        raise StopWordsNotFoundError(error)
    return stop_word

def _tokenizer(text):
    '''
    You can change the tokenizer function here
    '''
    return jieba.cut(text)

def _vectorizer(vocabulary=None):
    return CountVectorizer(
            tokenizer=TOKENIZER,
            stop_words=STOP_WORDS,
            vocabulary=vocabulary)

# Base Dir
CHERRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cherry')

# Dir to store data and stop words
DATA_DIR = os.path.join(CHERRY_DIR, 'data')

# stop words list
STOP_WORDS = _stop_word(DATA_DIR)

# tokenizer function
TOKENIZER = _tokenizer
