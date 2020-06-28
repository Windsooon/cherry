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
from .base import load_all, load_data, get_vectorizer, get_clf
from .exceptions import MethodNotFoundError

class Search:
    def __init__(self, model, parameters, language=None, preprocessing=None, categories=None, encoding=None, method=None,
            vectorizer=None, cv=None, n_jobs=None, vectorizer_method=None, clf=None, clf_method=None,
            x_data=None, y_data=None):
        '''
        1. Build pipeline
        2. Run RandomizedSearchCV or GridSearchCV
        3. Display the best score
        '''
        x_data, y_data, vectorizer, clf = load_all(
            model, language=language, preprocessing=preprocessing,
            categories=categories, encoding=encoding, vectorizer=vectorizer,
            vectorizer_method=vectorizer_method, clf=clf,
            clf_method=clf_method, x_data=x_data, y_data=y_data)
        self._search(vectorizer, clf, method, parameters, x_data, y_data, cv, n_jobs)

    def _search(self, vectorizer, clf, method, parameters, x_data, y_data, cv, n_jobs):
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        if method == 'RandomizedSearchCV':
            search_clf = RandomizedSearchCV(text_clf, parameters, cv=cv, n_jobs=n_jobs)
        elif method == 'GridSearchCV':
            search_clf = GridSearchCV(text_clf, parameters, cv=cv, n_jobs=n_jobs)
        else:
            error = 'We didn\'t support {0} method yet'.format(method)
            raise MethodNotFoundError(error)
        self.best_score(search_clf, parameters, x_data, y_data)

    def best_score(self, search_clf, parameters, x_test, y_test):
        search_clf = search_clf.fit(x_test, y_test)
        print('score is {0}'.format(search_clf.best_score_))
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, search_clf.best_params_[param_name]))
