# -*- coding: utf-8 -*-

"""
cherry.api
~~~~~~~~~~~~
This module implements the cherry API.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from .config import POSITIVE, BINARY_CLASSIFICATION
from .models import Result, Trainer


def classify(text, lan='Chinese', split=None):
    return _build_result(text=text, lan=lan, split=split)


def train(
        lan='Chinese', positive=POSITIVE,
        binary=BINARY_CLASSIFICATION, test=False):
    return Trainer(lan=lan, positive=positive, binary=binary)


def _build_result(**kwargs):
    '''
    Build classify result
    '''
    return Result(**kwargs)
