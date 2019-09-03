# -*- coding: utf-8 -*-
"""
cherry.gridsearch
~~~~~~~~~~~~~~~~~~~
This module contains the set of cheery' grid search.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from .base import load_data, get_vectorizer, get_clf
from .exceptions import MethodNotFoundError

class Search:
    def __init__(self, model, **kwargs):
        '''
        1. Build pipeline
        2. Run  RandomizedSearchCV or GridSearchCV
        3. Display the best score
        '''
        vectorizer = kwargs['vectorizer']
        vectorizer_method = kwargs['vectorizer_method']
        x_data = kwargs['x_data']
        y_data = kwargs['y_data']
        if not (x_data and y_data):
            x_data, y_data = load_data(model)
        clf = kwargs['clf']
        clf_method = kwargs['clf_method']
        if not vectorizer:
            vectorizer = get_vectorizer(model, vectorizer_method)
        if not clf:
            clf = get_clf(model, clf_method)
        parameters = kwargs['parameters']
        method = kwargs['method']
        cv = kwargs['cv']
        iid = kwargs['iid']
        n_jobs = kwargs['n_jobs']
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        if method == 'RandomizedSearchCV':
            search_clf = RandomizedSearchCV(text_clf, parameters, cv=cv, iid=iid, n_jobs=n_jobs)
        elif method == 'GridSearchCV':
            search_clf = GridSearchCV(text_clf, parameters, cv=cv, iid=iid, n_jobs=n_jobs)
        else:
            error = 'We didn\'t support {0} method yet'.format(method)
            raise MethodNotFoundError(error)
        self.best_score(search_clf, parameters, x_data, y_data)

    def best_score(self, search_clf, parameters, x_test, y_test):
        search_clf = search_clf.fit(x_test, y_test)
        print('score is {0}'.format(search_clf.best_score_))
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, search_clf.best_params_[param_name]))
