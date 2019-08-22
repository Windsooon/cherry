# -*- coding: utf-8 -*-

"""
cherry.classify
~~~~~~~~~~~~
This module implements the cherry classify.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""


import os
import pickle
from .config import DATA_DIR, _vectorizer
from .exceptions import CacheNotFoundError


class Classify:
    def __init__(self, **kwargs):
        self._load_cache()
        text = kwargs['text']
        self._res = self._classify(text)

    @property
    def get_word_list(self):
        return self.word_list

    @property
    def get_res(self):
        return self._res

    def _load_cache(self):
        '''
        Load cache from pre-trained model
        '''
        self.trained_model = self._load_from_file('trained.pkl')
        self.vocabulary = self._load_from_file('ve.pkl')

    def _load_from_file(self, filename):
        '''
        Load file from filename
        '''
        cache_path = os.path.join(DATA_DIR, filename)
        try:
            with open(cache_path, 'rb') as f:
                self.trained_model = pickle.load(f)
        except FileNotFoundError:
            error = (
                'Cache files not found,' +
                'maybe you should train the data first.')
            raise CacheNotFoundError(error)

    def _classify(self, text):
        '''
        1. Build vector using pre-trained vocabulary
        2. Transform the input text to text vector
        3. Predict the text
        '''
        self.vector = _vectorizer(vocabulary=self.vocabulary)
        text_vector = self.vector.transform(text)
        return self.trained_model.predict_prob(text_vector)
