# -*- coding: utf-8 -*-

"""
cherry.performance
~~~~~~~~~~~~
This module implements the cherry performance.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import operator
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn import metrics
from .base import load_data, write_file, get_vectorizer_and_clf, get_vectorizer, get_clf
from .trainer import Trainer
from .classifyer import Classify
from .exceptions import MethodNotFoundError

class Performance:
    def __init__(self, model, language=None, **kwargs):
        x_data = kwargs['x_data']
        y_data = kwargs['y_data']
        if not (x_data and y_data):
            cache = load_data(model)
            x_data, y_data = cache.data, cache.target
        kw_vectorizer = kwargs.get('vectorizer', None)
        kw_clf = kwargs.get('clf', None)
        vectorizer, clf = get_vectorizer_and_clf(
            language, kw_vectorizer, kw_clf,
            kwargs['vectorizer_method'], kwargs['clf_method'])
        n_splits = kwargs['n_splits']
        output = kwargs['output']
        # Use 
        for train_index, test_index in KFold(n_splits=n_splits, shuffle=True).split(y_data):
            x_train = operator.itemgetter(*train_index)(x_data)
            x_test = operator.itemgetter(*test_index)(x_data)
            y_train, y_test = y_data[train_index], y_data[test_index]
            print('Calculating score')
            self.score(vectorizer, clf, x_train, y_train, x_test, y_test, output)

    def score(self, vectorizer, clf, x_train, y_train, x_test, y_test, output):
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        text_clf.fit(x_train, y_train)
        predicted = text_clf.predict(x_test)
        report = metrics.classification_report(y_test, predicted)
        if output == 'Stdout':
            print(report)
        else:
            self.write_file(output, report)
