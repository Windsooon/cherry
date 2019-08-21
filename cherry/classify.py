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
from operator import itemgetter
import numpy as np
from .config import DATA_DIR, VECTORIZER
from .exceptions import CacheNotFoundError


class Classify:
    def __init__(self, **kwargs):
        self._load_cache()
        text = kwargs['text']
        self.percentage, self.word_list = self._classify(text)

    @property
    def get_percentage(self):
        return self.percentage

    @property
    def get_word_list(self):
        return self.word_list

    def _load_cache(self):
        cache_path = os.path.join(DATA_DIR, 'train.pkl')
        try:
            with open(cache_path, 'rb') as f:
                self.train_model = pickle.load(f)
        except FileNotFoundError:
            error = (
                'Cache files not found,' +
                'maybe you should train the data first.')
            raise CacheNotFoundError(error)
        cache_path = os.path.join(DATA_DIR, 've.pkl')
        try:
            with open(cache_path, 'rb') as f:
                self.ve = pickle.load(f)
        except FileNotFoundError:
            error = (
                'Cache files not found,' +
                'maybe you should train the data first.')
            raise CacheNotFoundError(error)

    def _classify(self, text):
        vector = self.ve.fit([text])
        breakpoint()
        return self.train_model.predict(vector)

    def _update_category(self, lst):
        '''
        Convert log to percentage
        '''
        # [('gamble.dat', -6.73...), ('normal.dat', -8.40...)]
        out_lst = [
            (self.CLASSIFY[i], lst[i]) for i in range(len(self.CLASSIFY))]
        # [('gamble.dat', -6.73...), ('normal.dat', -8.40...)]
        sorted_lst = sorted(out_lst, key=itemgetter(1), reverse=True)
        # [('gamble.dat', 1.0), ('normal.dat', 0.31...)]
        relative_lst = [(k, 2**(v-sorted_lst[0][1])) for k, v in sorted_lst]
        # [('gamble.dat', 0.76...), ('normal.dat', 0.23...)]
        percentage_lst = [
            (k, self._round(v/sum(v for _, v in relative_lst)))
            for k, v in relative_lst]
        return percentage_lst
