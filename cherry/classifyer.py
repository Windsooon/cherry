# -*- coding: utf-8 -*-

"""
cherry.classify
~~~~~~~~~~~~
This module implements the cherry classify.
:copyright: (c) 2018-2019 by Windson Yang :license: MIT License, see LICENSE for more details.
"""


import os
import numpy as np
from sklearn.exceptions import NotFittedError
from .base import load_cache
from .exceptions import TokenNotFoundError

# Only load cache once
CACHE = None

class Classify:
    def __init__(self, model, text=None):
        global CACHE
        if not text:
            error = 'Some of the tokens in text never appear in training data'
            raise TokenNotFoundError(error)
        if not CACHE:
            self.trained_model, self.vector = self._load_cache(model)
            CACHE = (self.trained_model, self.vector)
        else:
            self.trained_model, self.vector = CACHE
        self._classify(text)

    def get_word_list(self):
        word_list = []
        for tv in self.text_vector.toarray():
            word_list.append(sorted(
                [word for word in list(zip(tv, self.vector.get_feature_names())) if word[0] != 0.0], reverse=True))
        return word_list

    def get_probability(self):
        return self.trained_model.predict_proba(self.text_vector)

    def _load_cache(self, model):
        '''
        Load cache from pre-trained model
        '''
        trained_model = load_cache(model, 'clf.pkz')
        vector = load_cache(model, 've.pkz')
        return trained_model, vector

    def _classify(self, text):
        '''
        1. Build vector using pre-trained cache
        2. Transform the input text into text vector
        3. return the probability and word_list
        '''
        if isinstance(text, str):
            text = [text]
        try:
            self.text_vector = self.vector.transform(text)
        except NotFittedError:
            error = 'None of the word exist in training data'
            raise TokenNotFoundError(error)
