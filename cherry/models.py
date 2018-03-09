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
from .config import DATA_DIR
from .tokenizer import Token
from .exceptions import TestDataTooManyError


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
        self._test_data, self._train_data = [], []
        self.data_list, self.CLASSIFY = [], []
        self._read_files()
        # self.trainer = self._train_data(**kwargs)

    # @property
    # def get_status(self):
    #     return self.trainer.status

    @property
    def get_test_data(self):
        return self._test_data

    @property
    def get_train_data(self):
        return self._train_data

    def _train_data(self, **kwargs):
        test_data, train_data = self._split_data(kwargs['test'])

    def _split_data(self, test_num=None):
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
        test_num = test_num or self.data_len//5
        if test_num > self.data_len - 1:
            error = (
                'Test data numbers should small than {0}.'
                .format(self.data_len))
            raise TestDataTooManyError(error)

        random_list = random.sample(range(0, self.data_len), test_num)
        # Get test data
        for i in range(self.data_len):
            if i in random_list:
                self._test_data.append(self.data_list[i])
            else:
                self._train_data.append(self.data_list[i])

    def _read_files(self, lan):
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
        file_path = os.path.join(DATA_DIR, 'data/' + lan + '/data/')
        for i in range(len(file_path)):
            with open(file_path[i], encoding='utf-8') as f:
                for data in f.readlines():
                    self.data_list.append((i, data))
                # Get file name
                self.CLASSIFY.append(
                    os.path.basename(os.path.normpath(self.file_path[i])))
        self.data_len = len(self.classify)
