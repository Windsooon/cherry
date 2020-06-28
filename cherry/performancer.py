
"""
cherry.performance
~~~~~~~~~~~~
This module implements the cherry performance.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""
import os
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
        self.n_splits = n_splits
        self.output = output
        self.x_data, self.y_data, self.vectorizer, self.clf = load_all(
            model, language=language, preprocessing=preprocessing,
            categories=categories, encoding=encoding, vectorizer=vectorizer,
            vectorizer_method=vectorizer_method, clf=clf,
            clf_method=clf_method, x_data=x_data, y_data=y_data)

    def get_score(self):
        # TODO:  operator.itemgetter maybe is not the best solution
        print('Calculating score, depending on your datasets size, this may take several minutes to several hours.')
        for train_index, test_index in KFold(
                n_splits=self.n_splits, shuffle=True).split(self.y_data):
            x_train = operator.itemgetter(*train_index)(self.x_data)
            x_test = operator.itemgetter(*test_index)(self.x_data)
            y_train, y_test = self.y_data[train_index], self.y_data[test_index]
            self._score(self.vectorizer, self.clf, x_train, y_train, x_test, y_test, self.output)

    def _score(self, vectorizer, clf, x_train, y_train, x_test, y_test, output):
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        text_clf.fit(x_train, y_train)
        predicted = text_clf.predict(x_test)
        report = metrics.classification_report(y_test, predicted)
        if output == 'Stdout':
            for index, (input, prediction, label) in enumerate(zip (x_test, predicted, y_test)):
                if prediction != label:
                    print('Text:', input, 'has been classified as:', prediction, 'should be:', label)
            print(report)
        else:
            write_file(os.path.join(os.getcwd(), 'report'), report)
