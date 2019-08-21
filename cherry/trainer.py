# -*- coding: utf-8 -*-

"""
cherry.trainer
~~~~~~~~~~~~
This module implements the cherry Trainer.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import cPickle
import sklearn
import panda as pd

from .config import DATA_DIR, STOP_WORDS, TOKENIZER


class Trainer:
    def __init__(self, **kwargs):
        self._write_cache()
        self._read_data()
        self.train()
        self._write_cache()

    def train(self):
        '''
        Train bayes model with input data
        '''
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            tokenizer=TOKENIZER,
            stop_words=STOP_WORDS)
        training_data = vectorizer.fit_transform(self.x_data)
        naive_bayes = MultinomialNB()
        naive_bayes.fit(self.x_data, self.y_data)

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
        cache_path = os.path.join(DATA_DIR, '/bayes.pkl')
        with open(cache_path, 'wb') as f:
            cPickle.dump(naive_bayes, f)
