# -*- coding: utf-8 -*-

"""
cherry.api
~~~~~~~~~~~~
This module implements the cherry API.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from .base import load_data
from .config import PREFIX
from .trainer import Trainer
from .classify import Classify
from .performance import Performance
from .search import Search


def classify(text):
    '''
    return a Classify object which contains *probability* and *word_list*

    input: text (string): the text to be classified
    output: Classify (Classify object)
    '''
    return Classify(text=text)

def train(prefix=PREFIX, vectorizer=None, clf=None, x_data=None, y_data=None):
    '''
    Train the data inside data dir

    input prefix (string): prefix of the data file (i.e 'chinese_classify')
          vectorizer (BaseEstimator object): feature extraction method, should be CountVectorizer or TfidfVectorizer object
          clf (Classifier object): Classifier object, like DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier
          x_data (array): text to be trained
          y_data (array): text label to be trained
          clf (string): classify methore, should be 'MNB', 'RandomForest', 'AdaBoost' or 'SGD'
    '''
    if not (x_data and y_data):
        x_data, y_data = load_data(prefix)
    return Trainer(vectorizer=vectorizer, clf=clf, x_data=x_data, y_data=y_data)

def performance(prefix=PREFIX, vectorizer=None, clf=None, method='kfolds', n_splits=5, output='Stdout'):
    '''
    Calculate scores and ROC from the models
    '''
    return Performance(prefix=PREFIX, vectorizer=vectorizer, clf=clf, method=method, n_splits=n_splits, output=output)

def search(parameters, prefix=PREFIX, method='RandomizedSearchCV', cv=3, iid=False, n_jobs=1):
    '''
    Search the best parameters
    '''
    return Search(prefix=prefix, parameters=parameters, method=method, cv=cv, iid=iid, n_jobs=n_jobs)
