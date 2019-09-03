# -*- coding: utf-8 -*-

"""
cherry.performance
~~~~~~~~~~~~
This module implements the cherry performance.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn import metrics
from .base import load_data, write_file, get_vectorizer, get_clf
from .trainer import Trainer
from .classifyer import Classify
from .exceptions import MethodNotFoundError

class Performance:
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
        n_splits = kwargs['n_splits']
        output = kwargs['output']
        for train_index, test_index in KFold(n_splits=n_splits, shuffle=True).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            print('Calculating score')
            self.score(vectorizer, clf, x_train, y_train, x_test, y_test, output)

    def score(self, vectorizer, clf, x_train, y_train, x_test, y_test, output):
        vectorizer = DEFAULT_VECTORIZER if not vectorizer else vectorizer
        clf = DEFAULT_CLF if not clf else clf
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
