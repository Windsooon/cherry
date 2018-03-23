# -*- coding: utf-8 -*-

"""
cherry.config
~~~~~~~~~~~~
Base config of cherry classify
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import jieba
from nltk.tokenize import word_tokenize

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cherry')


def _jieba_cut(text, stop_word):
    return [
        t for t in jieba.cut(text) if len(t) > 1
        and t not in stop_word]


def _word_tokenize(text, stop_word):
    return [
        t for t in word_tokenize(text) if len(t) > 1
        and t not in stop_word]


LAN_DICT = {
    'Chinese': {
        'dir': False,
        'type': '.dat',
        'split': _jieba_cut},
    'English': {
        'dir': True,
        'type': '.txt',
        'split': _word_tokenize}}
