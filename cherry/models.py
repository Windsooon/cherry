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
import pickle
import numpy as np
from .config import DATA_DIR
from .tokenizer import Token
from .exceptions import TestDataNumError, CacheNotFoundError

np.set_printoptions(threshold=np.nan)


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
    def get_word_list(self):
        return self.word_list

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
        Bayes classify
        '''
        possibility_vector = []
        percentage_list = []
        for i in self._ps_vector:
            # final_vector: [0, -7.3, 0, 0, -8, ...]
            final_vector = i[0] * self.word_vec
            # word_index: [1, 4]
            word_index = np.nonzero(final_vector)
            non_zero_word = np.array(self._vocab_list)[word_index]
            # non_zero_vector: [-7.3, -8]
            non_zero_vector = final_vector[word_index]
            possibility_vector.append(non_zero_vector)
            percentage_list.append(sum(final_vector) + i[1])
        possibility_array = np.array(possibility_vector)
        max_val = max(percentage_list)
        for i, j in enumerate(percentage_list):
            if j == max_val:
                max_array = possibility_array[i, :]
                left_array = np.delete(possibility_array, i, 0)
                sub_array = np.zeros(max_array.shape)
                for k in left_array:
                    sub_array += max_array - k
                return self._clean_data(
                    percentage_list), \
                    list(zip(non_zero_word, sub_array))

    def _softmax(self, lst):
        '''
        Compute softmax values for each sets of scores in x.
        '''
        return np.exp(lst) / np.sum(np.exp(lst), axis=0)

    def _min_max(self, lst):
        '''
        Min-Max Normalization
        '''
        return [(x-min(lst))/(max(lst)-min(lst)) for x in lst]

    def _clean_data(self, lst):
        return list(zip(self.CLASSIFY, self._softmax(self._min_max(lst))))

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
        self._get_vocab_list()
        self.matrix_list = self._get_vocab_matrix()
        self._training()
        self._write_cache()

    @property
    def meta_classify(self):
        return self.CLASSIFY

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
        for k, data in self._train_data:
            token = Token(text=data, lan=self.lan, split=self.split)
            vocab_set = vocab_set | set(token.tokenizer)
        self._vocab_list = list(vocab_set)

    def _get_vocab_matrix(self):
        '''
        Convert all sentences to vector
        '''
        return [self._data_to_vector(i[1]) for i in self._train_data]

    def _data_to_vector(self, data):
        '''
        Convert strings to vector depends on vocal_list
        type data: strings
        '''
        return_vec = [0]*len(self._vocab_list)
        token = Token(text=data, lan=self.lan, split=self.split)
        for i in token.tokenizer:
            if i in self._vocab_list:
                return_vec[self._vocab_list.index(i)] += 1
        return return_vec

    def _training(self):
        '''
        Native bayes training
        '''
        vector = np.ones(len(self.matrix_list[0]))
        self._ps_vector = []
        vector_list = [{
            'vector': vector, 'cal': 2.0, 'num': 0.0}
            for i in range(len(self.CLASSIFY))]
        for k, v in enumerate(self.train_data):
            vector_list[v[0]]['num'] += 1
            vector_list[v[0]]['vector'] += self.matrix_list[k]
            vector_list[v[0]]['cal'] += sum(self.matrix_list[k])
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
