# -*- coding: utf-8 -*-

"""
cherry.trainer
~~~~~~~~~~~~
This module implements the cherry Trainer.
:copyright: (c) 2018-2020 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import pickle
from sklearn.pipeline import Pipeline
from .base import DATA_DIR, load_all, load_data, get_vectorizer_and_clf, \
    get_vectorizer, get_clf, write_cache
from .exceptions import *


class Trainer:
    def __init__(
            self, model, language=None, preprocessing=None, categories=None,
            encoding=None, vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=None, y_data=None):
        '''
        Data should be stored in a two levels folder structure like this:

        datasets/
          model_name/
            category1/
              file_1.txt
              file_2.txt
              file_42.txt
            category2/
              file_43.txt
              file_44.txt
        '''
        x_data, y_data, vectorizer, clf = load_all(
            model, language=language, preprocessing=preprocessing,
            categories=categories, encoding=encoding, vectorizer=vectorizer,
            vectorizer_method=vectorizer_method, clf=clf,
            clf_method=clf_method, x_data=x_data, y_data=y_data)
        Trainer._train(vectorizer, clf, x_data, y_data)
        # TODO: If the cache files existed, ask user to comfirm overwrite.
        write_cache(model, vectorizer, 've.pkz')
        write_cache(model, clf, 'clf.pkz')

    @classmethod
    def _train(cls, vectorizer, clf, x_data, y_data):
        '''
        Train bayes model with input data and decide which feature extraction method
        and classify method should use
        '''
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        print('Training data, depending on your datasets size, this may take several minutes to several hours.')
        text_clf.fit(x_data, y_data)
