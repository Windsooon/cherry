# -*- coding: utf-8 -*-

"""
cherry.classify
~~~~~~~~~~~~
This module implements the cherry classify.
:copyright: (c) 2018-2019 by Windson Yang :license: MIT License, see LICENSE for more details.
"""


import os
import numpy as np
from .base import load_cache


class Classify:
    def __init__(self, **kwargs):
        text = kwargs['text']
        N = kwargs['N']
        self._load_cache()
        self.probability, self.word_list = self._classify(text, N)

    @property
    def get_word_list(self):
        return self.word_list

    @property
    def get_probability(self):
        return self.probability

    def _load_cache(self):
        '''
        Load cache from pre-trained model
        '''
        self.trained_model = load_cache('trained.pkl')
        self.vector = load_cache('ve.pkl')

    def _classify(self, text, N):
        '''
        1. Build vector using pre-trained vocabulary
        2. Transform the input text to text vector
        3. return the probability and word_list of the text
        '''
        text_vector = self.vector.transform(text)
        tv = text_vector.toarray()[0, :]
        feature_names = np.asarray(self.vector.get_feature_names())
        word_list = sorted([word for word in list(zip(tv, feature_names)) if word[0] != 0.0][:N], reverse=True)
        probability = self.trained_model.predict_proba(text_vector)
        return probability, word_list
