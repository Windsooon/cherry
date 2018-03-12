# -*- coding: utf-8 -*-

"""
cherry.models
~~~~~~~~~~~~
This module implements the cherry models.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import random
import numpy as np
from .config import DATA_DIR
from .tokenizer import Token
from .exceptions import TestDataNumError


class Result:
    def __init__(self, **kwargs):
        self.token = Token(**kwargs)

    def _calculate_ps(self, text):
        pass

    @property
    def get_token(self):
        return self.token


class Trainer:
    def __init__(self, **kwargs):
        self.data_list, self.CLASSIFY = [], []
        self.lan = kwargs['lan']
        self.split = kwargs['split']
        self.test_mode = kwargs['test_mode']
        # Get test data num when test_mode is true
        if self.test_mode:
            self._test_num = kwargs['test_num']
        else:
            self._test_num = 0

        # Get data from data directory
        self._read_files()
        # Split data by test num
        self._split_data(self.test_num)
        self._vocab_list()
        self.matrix_list = self._get_vocab_matrix()

    @property
    def test_num(self):
        return self._test_num

    @property
    def test_data(self):
        return self._test_data

    @property
    def train_data(self):
        return self._train_data

    @property
    def ps_vector(self):
        return self._ps_vector

    @test_num.setter
    def set_test_num(self, num):
        if num >= self.data_len or num < 0:
            error = (
                'Test data numbers should between data length and zero.'
                .format(self.data_len))
            raise TestDataNumError(error)
        self._test_num = num

    def _split_data(self, test_num):
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
        random_list = random.sample(range(0, self.data_len), test_num)
        # Get test data
        self._test_data, self._train_data = [], []
        for i in range(self.data_len):
            if i in random_list:
                self._test_data.append(self.data_list[i])
            else:
                self._train_data.append(self.data_list[i])

    def _read_files(self):
        '''
        Read data from given file path

        :param file_path: ./data/Chinese/data/

        data_list:
            [
                (0, "What a lovely day"),
                (1, "I like gambling"),
                (0, "I love my dog sunkist)"
            ]
        self.CLASSIFY: ['gamble.dat', 'normal.dat']
        self.data_len: 3
        '''
        file_dir_path = os.path.join(DATA_DIR, 'data/' + self.lan + '/data/')
        file_path = [
            os.path.join(file_dir_path, f) for f in
            os.listdir(file_dir_path) if f.endswith('.dat')]
        for i in range(len(file_path)):
            with open(file_path[i], encoding='utf-8') as f:
                for data in f.readlines():
                    self.data_list.append((i, data))
                # Get file name
                self.CLASSIFY.append(
                    os.path.basename(os.path.normpath(file_path[i])))
        self.data_len = len(self.data_list)

    def _vocab_list(self):
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
        for k, data in self._train_data:
            token = Token(text=data, lan=self.lan, split=self.split)
            vocab_set = vocab_set | set(token.tokenizer)
        self.vocab_list = list(vocab_set)

    def _get_vocab_matrix(self):
        '''
        Convert all sentences to vector
        '''
        print(self._train_data)
        return [self._data_to_vector(i) for i in self._train_data[1]]

    def _data_to_vector(self, data):
        '''
        Convert strings to vector depends on vocal_list
        type data: strings
        '''
        return_vec = [0]*len(self.vocab_list)
        token = Token(text=data[1], lan=self.lan, split=self.split)
        for i in token.tokenizer:
            if i in self.vocab_list:
                return_vec[self.vocab_list.index(i)] += 1
        return return_vec

    def _training(self):
        '''
        Native bayes training
        '''
        num = np.ones(len(self.matrix_list[0]))
        self._ps_vector = []
        start_index = 0
        for k, v in enumerate(self.train_data):
            cal, sentence = 2.0, 0.0
            if v[0] == start_index:
                sentence += 1
                num += self.matrix_list[k]
                cal += sum(self.matrix_list[k])
            else:
                self._ps_vector.append(
                    np.log(num/cal), sentence/len(self.train_data))
                start_index += 1
