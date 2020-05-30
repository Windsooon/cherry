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
from .base import DATA_DIR, load_data, get_vectorizer, get_clf, write_cache
from .exceptions import *


class Trainer:
    def __init__(self, model, language=None, categories=None, encoding=None, **kwargs):
        '''
        Data should be stored in a two levels folder structure such as the following:

        dataset/
          model_name/
            category1/
              file_1.txt file_2.txt … file_42.txt
            category2/
              file_43.txt file_44.txt …
        '''
        try:
            cache = load_data(model, categories=categories, encoding=encoding)
        except FilesNotFoundError:
            error = 'Please make sure your put the {0} data inside `dataset` folder or choose models inside BUILD_IN_MODELS.'.format(model)
            raise FilesNotFoundError(error)
        vectorizer = kwargs.get('vectorizer', None)
        # By default, Cherry will use CountVectorizer() if `vectorizer` is None
        vectorizer_method = kwargs.get('vectorizer_method', None)
        if not vectorizer:
            vectorizer = get_vectorizer(model, language, vectorizer_method)
        # By default, Cherry will use CountVectorizer() if `clf` is None
        clf = kwargs.get('clf', None)
        clf_method = kwargs.get('clf_method', None)
        if not clf:
            clf = get_clf(model, clf_method)
        # Start training data
        self.train(language, vectorizer, clf, cache)
        # TODO: If the cache files existed, ask user to comfirm.
        write_cache(model, vectorizer, 've.pkz')
        write_cache(model, clf, 'clf.pkz')

    def train(self, vectorizer, clf, cache):
        '''
        Train bayes model with input data and decide which feature extraction method
        and classify method should use
        '''
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        print('Training may take some time depending on your dataset')
        text_clf.fit(cache.data, cache.target)
