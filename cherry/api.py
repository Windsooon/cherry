# -*- coding: utf-8 -*-

"""
cherry.api
~~~~~~~~~~~~
This module implements the cherry API.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from .trainer import Trainer
from .classify import Result


def classify(text, lan='Chinese', split=None):
    return Result(text=text, lan=lan, split=split)


def train(
        lan='Chinese', test_num=0, split=None):
    return Trainer(
        lan=lan, test_num=test_num, split=split)
