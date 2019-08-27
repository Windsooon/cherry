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
from sklearn.pipeline import Pipeline
from .config import DATA_DIR, tfidf_vectorizer, count_vectorizer, get_vect_and_clf, read_data
from .exceptions import MethodNotFoundError


class Trainer:
    def __init__(self, **kwargs):
        x_data = kwargs['x_data']
        y_data = kwargs['y_data']
        feature = kwargs['feature']
        clf = kwargs['clf']
        self.train(x_data, y_data, feature, clf)
        self._write_cache()

    def train(self, x_data, y_data, feature='Count', ngram_range=(1, 1), max_df=1.0, min_df=1, clf='MNB'):
        '''
        Train bayes model with input data and decide which feature extraction method
        and classify method should use
        '''
        self.vectorizer, self.clf = get_vect_and_clf(feature, clf)
        text_clf = Pipeline([
            ('vectorizer', self.vectorizer),
            ('clf', self.clf)])
        text_clf.fit(x_data, y_data)

    def _write_cache(self):
        '''
        Write cache file under DATA_DIR
        '''
        cache_path = os.path.join(DATA_DIR, 'trained.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.clf, f)
        cache_path = os.path.join(DATA_DIR, 've.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
