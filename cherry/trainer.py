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
            error = ('Please make sure your put the {0} data inside `dataset` '
                    'folder or use model inside BUILD_IN_MODELS.'.format(model))
            raise FilesNotFoundError(error)
        vectorizer, clf = Trainer._get_vectorizer_and_clf(language, kwargs)
        Trainer._train(vectorizer, clf, cache)
        # TODO: If the cache files existed, ask user to comfirm overwrite.
        write_cache(model, vectorizer, 've.pkz')
        write_cache(model, clf, 'clf.pkz')

    @classmethod
    def _get_vectorizer_and_clf(cls, language, kwargs):
        vectorizer = kwargs.get('vectorizer', None)
        if not vectorizer:
            vectorizer_method = kwargs.get('vectorizer_method')
            vectorizer = get_vectorizer(language, vectorizer_method)
        clf = kwargs.get('clf', None)
        if not clf:
            clf_method = kwargs.get('clf_method')
            clf = get_clf(clf_method)
        return vectorizer, clf

    @classmethod
    def _train(cls, vectorizer, clf, cache):
        '''
        Train bayes model with input data and decide which feature extraction method
        and classify method should use
        '''
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        print('Depending on your dataset, training may take several minutes to several hours.')
        text_clf.fit(cache.data, cache.target)
