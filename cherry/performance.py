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
from sklearn.naive_bayes import MultinomialNB
from .config import read_data, tfidf_vectorizer, count_vectorizer
from .trainer import Trainer
from .classify import Classify
from .exceptions import MethodNotFoundError

class Performance:
    def __init__(self, **kwargs):
        method = kwargs['method']
        n_splits = kwargs['n_splits']
        output = kwargs['output']
        x_train, x_test, y_train, y_test = self.split_data(method, n_splits)
        self.score(x_train, x_test, y_train, y_test, output)

    def split_data(self, method, n_splits):
        '''
        TODO
        '''
        if method == 'kfolds':
            x_data, y_data = read_data()
            cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
            for train_index, test_index in cv.split(x_data):
                x_train, x_test = x_data[train_index], x_data[test_index]
                y_train, y_test = y_data[train_index], y_data[test_index]
                return x_train, x_test, y_train, y_test
        elif method == 'leaveone':
            pass
        else:
            error = 'We didn\'t support this method yet'
            raise MethodNotFoundError(error)

    def score(self, x_train, x_test, y_train, y_test, output):
        text_clf = Pipeline([
            ('tfidf', count_vectorizer()),
            ('clf', MultinomialNB())])
        text_clf.fit(x_train, y_train)
        predicted = text_clf.predict(x_test)
        report = metrics.classification_report(y_test, predicted)
        if output == 'Stdout':
            print(report)
        else:
            self.write_to_file(output, report)
