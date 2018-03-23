# -*- coding: utf-8 -*-

"""
cherry.classify
~~~~~~~~~~~~
This module implements the cherry classify.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""


import os
import pickle
from operator import itemgetter
import numpy as np
from .config import DATA_DIR
from .tokenizer import Token
from .exceptions import CacheNotFoundError


class Result:
    def __init__(self, **kwargs):
        self.token = Token(**kwargs)
        self.lan = kwargs['lan']
        self._load_cache()
        self._data_to_vector()
        self.percentage, self.word_list = self._bayes_classify()

    @property
    def get_percentage(self):
        return self.percentage

    @property
    def get_token(self):
        return self.token.tokenizer

    @property
    def get_word_list(self):
        return self.word_list

    def _round(self, val):
        return float(np.around(val, decimals=3))

    def _load_cache(self):
        cache_path = os.path.join(DATA_DIR, 'data/' + self.lan + '/cache/')
        try:
            with open(cache_path + 'vocab_list.cache', 'rb') as f:
                self._vocab_list = pickle.load(f)
            with open(cache_path + 'vector.cache', 'rb') as f:
                self._ps_vector = pickle.load(f)
            with open(cache_path + 'classify.cache', 'rb') as f:
                self.CLASSIFY = pickle.load(f)
        except FileNotFoundError:
            error = (
                'Cache files not found,' +
                'maybe you should train the data first.')
            raise CacheNotFoundError(error)

    def _data_to_vector(self):
        '''
        Convert input data to word_vector
        '''
        self.word_vec = [0]*len(self._vocab_list)
        for i in self.token.tokenizer:
            if i in self._vocab_list:
                self.word_vec[self._vocab_list.index(i)] += 1

    def _bayes_classify(self):
        '''
        Calculate the probability of different category
        '''
        possibility_vector = []
        log_list = []
        # self._ps_vector: ([-3.44, -3.56, -2.90], 0.4)
        for i in self._ps_vector:
            # final_vector: [0, -7.3, 0, 0, -8, ...]
            final_vector = i[0] * self.word_vec
            # Get most distance non zero word list
            word_index = np.nonzero(final_vector)
            non_zero_word = np.array(self._vocab_list)[word_index]
            # non_zero_vector: [-7.3, -8]
            non_zero_vector = final_vector[word_index]
            possibility_vector.append(non_zero_vector)
            log_list.append(sum(final_vector) + i[1])
        possibility_array = np.array(possibility_vector)
        max_val = max(log_list)
        for i, j in enumerate(log_list):
            if j == max_val:
                max_array = possibility_array[i, :]
                left_array = np.delete(possibility_array, i, 0)
                sub_array = np.zeros(max_array.shape)
                for k in left_array:
                    sub_array += max_array - k
                return self._update_category(log_list), \
                    sorted(
                        list(zip(non_zero_word, sub_array)),
                        key=lambda x: x[1], reverse=True)

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
