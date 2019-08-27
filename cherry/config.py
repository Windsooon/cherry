# -*- coding: utf-8 -*-

"""
cherry.config
~~~~~~~~~~~~
Base config of cherry classify
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import csv
import jieba
import pandas
from .exceptions import StopWordsNotFoundError, UnicodeFileEncodeError
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

# Base Dir
CHERRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cherry')

# Dir to store data and stop words
DATA_DIR = os.path.join(CHERRY_DIR, 'data')

def _stop_words(prefix):
    '''
    Return Stop words depent on prefix
    _stop_word('chinese_classify_') will return the data in
    DTAT_DIR/chinese_classify_stop_words.dat
    '''
    try:
        stop_words_path = os.path.join(DATA_DIR, prefix + 'stop_words.dat')
        with open(stop_word_path, encoding='utf-8') as f:
            stop_words = [l[:-1] for l in f.readlines()]
    except IOError:
        error = 'Stop word file not found'
        raise StopWordsNotFoundError(error)
    except UnicodeEncodeError as e:
        error = e
        raise UnicodeFileEncodeError(error)
    return stop_words

def load_data(prefix):
    df = pandas.read_csv(os.path.join(DATA_DIR, prefix + '_data.csv'))
    stop_words = _stop_words(prefix)
    return df['text'], df['label'], stop_words

def load_chinese_classify_data():
    '''
    Read data from data file inside DATA_DIR
    '''
    prefix = 'chinese_classify'
    return load_data(prefix)

def _tokenizer(text):
    '''
    You can change the tokenizer function here
    '''
    return [
        t for t in jieba.cut(text) if len(t) > 1
        and t not in STOP_WORDS]

def tfidf_vectorizer(vocabulary=None):
    return TfidfVectorizer(
            tokenizer=TOKENIZER,
            stop_words=STOP_WORDS,
            vocabulary=vocabulary,
            use_idf=True)

def count_vectorizer(ngram_range, max_df, min_df, vocabulary=None):
    return CountVectorizer(
            tokenizer=TOKENIZER,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            stop_words=STOP_WORDS,
            vocabulary=vocabulary)

def get_vect_and_clf(feature, ngram_range, max_df, min_df, clf):
    '''
    TODO
    '''
    if feature == 'Count':
        vectorizer = count_vectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    elif feature == 'Tfidf':
        vectorizer = tfidf_vectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    else:
        error = 'We didn\'t support {} feature extraction method.'.format(feature)
        raise MethodNotFoundError(error)
    mapping = {
        'MNB': MultinomialNB(), 'SGD': SGDClassifier(),
        'RF': RandomForestClassifier(), 'Ada': AdaBoostClassifier()
    }
    try:
        clf = mapping[clf]
    except KeyError:
        error = ' We didn\'t support {} classifier method.'.format(clf)
        raise MethodNotFoundError(error)
    return vectorizer, clf

def write_file(self, path, data):
    '''
    Write data to path
    '''
    with open(output, 'w') as f:
        f.write(report)
