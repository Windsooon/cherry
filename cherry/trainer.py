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
from .base import DATA_DIR, load_data, get_vectorizer, get_clf


class Trainer:
    def __init__(self, model, **kwargs):
        x_data = kwargs['x_data']
        y_data = kwargs['y_data']
        if not (x_data and y_data):
            x_data, y_data = load_data(model)
        vectorizer = kwargs['vectorizer']
        vectorizer_method = kwargs['vectorizer_method']
        clf = kwargs['clf']
        clf_method = kwargs['clf_method']
        if not vectorizer:
            vectorizer = get_vectorizer(model, vectorizer_method)
        if not clf:
            clf = get_clf(model, clf_method)
        self.train(vectorizer, clf, x_data, y_data)
        self._write_cache(model, vectorizer, clf)

    def train(self, vectorizer, clf, x_data, y_data):
        '''
        Train bayes model with input data and decide which feature extraction method
        and classify method should use
        '''
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        print('Training may take some time depending on your dataset')
        text_clf.fit(x_data, y_data)

    def _write_cache(self, model, vectorizer, clf):
        '''
        Write cache file under DATA_DIR
        '''
        cache_path = os.path.join(DATA_DIR, model + '/trained.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(clf, f)
        cache_path = os.path.join(DATA_DIR, model + '/ve.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(vectorizer, f)
