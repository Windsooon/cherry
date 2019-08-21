# -*- coding: utf-8 -*-

"""
cherry.trainer
~~~~~~~~~~~~
This module implements the cherry Trainer.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from .config import DATA_DIR, STOP_WORDS, TOKENIZER, VECTORIZER


class Trainer:
    def __init__(self, **kwargs):
        self._read_data()
        self.train()
        self._write_cache()

    def train(self):
        '''
        Train bayes model with input data
        '''
        self.ve = VECTORIZER.fit(self.x_data)
        breakpoint()
        training_data = self.ve.transform(self.x_data)
        self.bayes = MultinomialNB()
        self.bayes.fit(training_data, self.y_data)

    def _read_data(self):
        '''
        Read data from data file inside DATA_DIR
        '''
        # df = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'), skipinitialspace=True)
        df = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'))
        self.x_data = df['text']
        self.y_data = df['label']

    def _write_cache(self):
        '''
        Write cache file under DATA_DIR
        '''
        cache_path = os.path.join(DATA_DIR, 'train.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.bayes, f)
        cache_path = os.path.join(DATA_DIR, 've.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.ve, f)
