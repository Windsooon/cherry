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
from .analysis import Analysis
from .infomation import Info


def classify(text):
    '''
    return a Result object which contains *percentage* and *word_list*

    input: text (string): the text to be classified
    output: result (Result object)
    '''
    return Classify(text=text)

def train():
    '''
    Train the data inthe data dir with stop_word and split function

    input: stop_word (list) list of stop word (string)
           split (function) the function use to tokenizer the text in the data
    output: None

    '''
    return Trainer()

def analysis():
    '''
    Use k-fold cross validation t calculate precision, recall, F1 score and ROC canvas.
    '''
    return Analysis()
