# -*- coding: utf-8 -*-

"""
cherry.api
~~~~~~~~~~~~
This module implements the cherry API.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from .trainer import Trainer
from .classify import Classify
from .performance import Performance
from .config import read_data


def classify(text):
    '''
    return a Result object which contains *percentage* and *word_list*

    input: text (string): the text to be classified
    output: result (Result object)
    '''
    return Classify(text=text)

def train(x_data=None, y_data=None, feature='Count', clf='MNB'):
    '''
    Train the data inside data dir
    '''
    if not (x_data and y_data):
        x_data, y_data = read_data()
    return Trainer(x_data=x_data, y_data=y_data, feature=feature, clf=clf)

def performance(method='kfolds', n_splits=5):
    '''
    Calculate scores and ROC from the models
    '''
    return Performance(method=method, n_splits=n_splits)
