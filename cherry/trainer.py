# -*- coding: utf-8 -*-

"""
cherry.models
~~~~~~~~~~~~
This module implements the cherry Trainer.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import random
import pickle
import numpy as np
from .config import DATA_DIR, LAN_DICT
from .tokenizer import Token
from .infomation import Info
from .exceptions import TestDataNumError


class Trainer:
    def __init__(self, **kwargs):
        self.lan = kwargs['lan']
        self.split = LAN_DICT[self.lan]['split']
        self.dir = LAN_DICT[self.lan]['dir']
        self.type = LAN_DICT[self.lan]['type']
        # Get all data from data directory
        self.data_list, self.CLASSIFY = Info.read_files(
                lan=self.lan)
        self.data_len = len(self.data_list)
        self._test_num = kwargs['test_num']
        # Split data to train_data and test_data by test num
        self._split_data()
        self._get_vocab_list()
        self._get_vocab_matrix()
        self._training()
        self._write_cache()

    @property
    def vocab_list(self):
        return self._vocab_list

    @property
    def test_data_classify(self):
        return [k for k, v in self._test_data]

    @property
    def test_data(self):
        return self._test_data

    @property
    def train_data(self):
        return self._train_data

    @property
    def ps_vector(self):
        return self._ps_vector

    @property
    def test_num(self):
        return self._test_num

    @test_num.setter
    def test_num(self, num):
        if num >= self.data_len or num < 0:
            error = (
                'Test data numbers should between data length and zero.'
                .format(self.data_len))
            raise TestDataNumError(error)
        self._test_num = num

    def _split_data(self):
        '''
        Split data into test data and train data randomly.

        self.test_data:
            [
                (0, "What a lovely day"),
            ]
        self.train_data:
            [
                (1, "I like gambling"),
                (0, "I love my dog sunkist")
            ]
        '''
        random_list = random.sample(range(0, self.data_len), self._test_num)
        # Get test data
        self._test_data, self._train_data = [], []
        for i in range(self.data_len):
            if i in random_list:
                self._test_data.append(self.data_list[i])
            else:
                self._train_data.append(self.data_list[i])

    def _get_vocab_list(self):
        '''
        Get a list contain all unique non stop words belongs to train_data
        Set up:
        self.vocab_list:
            [
                'What', 'lovely', 'day',
                'like', 'gamble', 'love', 'dog', 'sunkist'
            ]
        '''
        vocab_set = set()
        all_train_data = ''.join([v for _, v in self._train_data])
        token = Token(text=all_train_data, lan=self.lan, split=self.split)
        vocab_set = vocab_set | set(token.tokenizer)
        self._vocab_list = list(vocab_set)

    def _get_vocab_matrix(self):
        '''
        Convert strings to vector depends on vocal_list
        '''
        array_list = []
        for k, data in self._train_data:
            return_vec = np.zeros(len(self._vocab_list))
            token = Token(text=data, lan=self.lan, split=self.split)
            for i in token.tokenizer:
                if i in self._vocab_list:
                    return_vec[self._vocab_list.index(i)] += 1
            array_list.append(return_vec)
        self._matrix_lst = array_list

    def _training(self):
        '''
        Native bayes training
        '''
        self._ps_vector = []
        vector_list = [{
            'vector': np.ones(len(self._matrix_lst[0])),
            'cal': 2.0, 'num': 0.0} for i in range(len(self.CLASSIFY))]
        for k, v in enumerate(self.train_data):
            vector_list[v[0]]['num'] += 1
            vector_list[v[0]]['vector'] += self._matrix_lst[k]
            vector_list[v[0]]['cal'] += sum(self._matrix_lst[k])
        for i in range(len(self.CLASSIFY)):
            self._ps_vector.append((
                np.log(vector_list[i]['vector']/vector_list[i]['cal']),
                np.log(vector_list[i]['num']/len(self.train_data))))

    def _write_cache(self):
        cache_path = os.path.join(DATA_DIR, 'data/' + self.lan + '/cache/')
        with open(cache_path + 'vocab_list.cache', 'wb') as f:
            pickle.dump(self._vocab_list, f)
        with open(cache_path + 'vector.cache', 'wb') as f:
            pickle.dump(self._ps_vector, f)
        with open(cache_path + 'classify.cache', 'wb') as f:
            pickle.dump(self.CLASSIFY, f)
