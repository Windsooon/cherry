# -*- coding: utf-8 -*-

"""
cherry.api
~~~~~~~~~~~~
This module implements the cherry API.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from .models import Result, Trainer


def classify(text, lan='Chinese', split=None):
    return Result(text=text, lan=lan, split=split)


def train(
        lan='Chinese', test_num=0,
        test_mode=False, split=None):
    return Trainer(
        lan=lan, test_num=test_num, test_mode=test_mode,
        split=split)
