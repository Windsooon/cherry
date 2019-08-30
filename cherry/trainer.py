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
from .base import DATA_DIR, load_data
from .config import DEFAULT_CLF, DEFAULT_VECTORIZER
from .exceptions import MethodNotFoundError


class Trainer:
    def __init__(self, **kwargs):
        x_data = kwargs['x_data']
        y_data = kwargs['y_data']
        vectorizer = kwargs['vectorizer']
        clf = kwargs['clf']
        vectorizer, clf = self.train(vectorizer, clf, x_data, y_data)
        self._write_cache(vectorizer, clf)

    def train(self, vectorizer, clf, x_data, y_data):
        '''
        Train bayes model with input data and decide which feature extraction method
        and classify method should use
        '''
        vectorizer = DEFAULT_VECTORIZER if not vectorizer else vectorizer
        clf = DEFAULT_CLF if not clf else clf
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        text_clf.fit(x_data, y_data)
        return vectorizer, clf

    def _write_cache(self, vectorizer, clf):
        '''
        Write cache file under DATA_DIR
        '''
        cache_path = os.path.join(DATA_DIR, 'trained.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(clf, f)
        cache_path = os.path.join(DATA_DIR, 've.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(vectorizer, f)
