# -*- coding: utf-8 -*-

"""
cherry.performance
~~~~~~~~~~~~
This module implements the cherry performance.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from sklearn.model_selection import KFold
from .config import read_data
from .trainer import Trainer
from .exceptions import MethodNotFoundError

class Performance:
    def __init__(self, **kwargs):
        method = kwargs['method']
        n_splits = kwargs['n_splits']
        self.split_data(method, n_splits)

    def split_data(self, method, n_splits):
        '''
        TODO
        '''
        if method == 'kfolds':
            x_data, y_data = read_data()
            cv = KFold(n_splits=n_splits, random_state=42, shuffle=False)
            for train_index, test_index in cv.split(x_data):
                x_train, x_test = x_data[train_index], x_data[test_index]
                y_train, y_test = y_data[train_index], y_data[test_index]
                Trainer(x_data=x_train, y_data=y_train, prefix='vali_')
        elif method == 'leaveone':
            pass
        else:
            error = 'We didn\'t support this method yet'
            raise MethodNotFoundError(error)

