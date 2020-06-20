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
from .base import load_all, load_data, write_file, get_vectorizer_and_clf, get_vectorizer, get_clf
from .trainer import Trainer
from .classifyer import Classify
from .exceptions import MethodNotFoundError

class Performance:
    def __init__(self, model, language=None, preprocessing=None, categories=None, encoding=None, vectorizer=None,
            n_splits=10, output=None, vectorizer_method=None, clf=None, clf_method=None, x_data=None, y_data=None):
        x_data, y_data, vectorizer, clf = load_all(
            model, language=language, preprocessing=preprocessing,
            categories=categories, encoding=encoding, vectorizer=vectorizer,
            vectorizer_method=vectorizer_method, clf=clf,
            clf_method=clf_method, x_data=x_data, y_data=y_data)
        # TODO:  operator.itemgetter maybe is not the best solution
        for train_index, test_index in KFold(n_splits=n_splits, shuffle=True).split(y_data):
            x_train = operator.itemgetter(*train_index)(x_data)
            x_test = operator.itemgetter(*test_index)(x_data)
            y_train, y_test = y_data[train_index], y_data[test_index]
            print('Calculating score, depending on your dataset size, this may take several minutes to several hours.')
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
