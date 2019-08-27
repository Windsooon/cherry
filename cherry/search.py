# -*- coding: utf-8 -*-
"""
cherry.gridsearch
~~~~~~~~~~~~~~~~~~~
This module contains the set of cheery' grid search.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

from .config import get_vect_and_clf

class Search:
    def __init__(self, prefix, parameters, method='RandomizedSearchCV', cv=3, iid=False, n_jobs=1):
        vectorizer_lst = [token, tfidf]

        text_clf = Pipeline([
            ('vectorizer', parameters['vectorizer']),
            ('clf', parameters['clf'])])
        del parameters['vectorizer']
        del parameters['clf']
        if method == 'RandomizedSearch':
            search_clf = RandomizedSearchCV(text_clf, parameters, cv=cv, iid=iid, n_jobs=n_jobs)
        elif method == 'GridSearch':
            search_clf = GridSearchCV(text_clf, parameters, cv=cv, iid=iid, n_jobs=n_jobs)
        else:
            raise
        x_test, y_test = load_data(prefix)
        search_clf = search_clf.fit(x_test, y_test)
        print(search_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
