# -*- coding: utf-8 -*-

"""
cherry.api
~~~~~~~~~~~~
This module implements the cherry API.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from .models import Result


def _build_result(**kwargs):
    '''
    Build classify result
    '''
    result = Result(**kwargs)
    return result


def classify(text, lan, split=None):
    return _build_result(text=text, lan=lan, split=split)
