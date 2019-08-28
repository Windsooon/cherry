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
from .base import load_data, write_file
from .config import DEFAULT_CLF, DEFAULT_VECTORIZER
from .trainer import Trainer
from .classify import Classify
from .exceptions import MethodNotFoundError

class Performance:
    def __init__(self, **kwargs):
        vectorizer = kwargs['vectorizer']
        clf = kwargs['clf']
        method = kwargs['method']
        n_splits = kwargs['n_splits']
        output = kwargs['output']
        prefix = kwargs['prefix']
        x_train, y_train = load_data(prefix)
        x_test, y_test = load_data(prefix + '_test')
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
