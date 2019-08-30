# -*- coding: utf-8 -*-

"""
cherry.api
~~~~~~~~~~~~
This module implements the cherry API.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from .base import load_data
from .config import FILENAME
from .trainer import Trainer
from .classifyer import Classify
from .performance import Performance
from .search import Search
from .display import Display


def classify(text, N=20):
    '''
    Return a Classify object which contains *probability* and *word_list*

    input: text (list of string): the text to be classified
    input: number of word list (int): how many word should be list in the word list
    output: Classify (Classify object)

    >>> cherry.classify(['Test string'])
    '''
    return Classify(text=text, N=N)

def train(filename=FILENAME, vectorizer=None, clf=None, x_data=None, y_data=None):
    '''
    Train the data inside data dir

    input filename (string): file name of the data file (i.e 'chinese_classify.dat')
          vectorizer (BaseEstimator object): feature extraction method, should be CountVectorizer or TfidfVectorizer object
          clf (Classifier object): Classifier object, like DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier
          x_data (array): text to be trained
          y_data (array): text label to be trained
          clf (string): classify methore, should be 'MNB', 'RandomForest', 'AdaBoost' or 'SGD'

    >>> cherry.train()
    '''
    if not (x_data and y_data):
        x_data, y_data = load_data(filename)
    return Trainer(vectorizer=vectorizer, clf=clf, x_data=x_data, y_data=y_data)

def performance(filename=FILENAME, vectorizer=None, clf=None, method='kfolds', n_splits=5, output='Stdout'):
    '''
    Calculate scores and ROC from the models

    >>> cherry.performance()
    '''
    return Performance(filename=filename, vectorizer=vectorizer, clf=clf, method=method, n_splits=n_splits, output=output)

def search(parameters, filename=FILENAME, method='RandomizedSearchCV', cv=3, iid=False, n_jobs=1):
    '''
    Search the best parameters

    >>> cherry.search()
    '''
    return Search(filename=FILENAME, parameters=parameters, method=method, cv=cv, iid=iid, n_jobs=n_jobs)

def display(vectorizer=None, clf=None, x_data=None, y_data=None, filename=FILENAME):
    '''
    Display the learning curve
    '''
    from .config import DEFAULT_VECTORIZER, DEFAULT_CLF
    if not vectorizer and not clf:
        vectorizer, clf = DEFAULT_VECTORIZER, DEFAULT_CLF
    if not (x_data and y_data):
        x_data, y_data = load_data(filename)
    return Display(vectorizer=vectorizer, clf=clf, x_data=x_data, y_data=y_data)
