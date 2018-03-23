# -*- coding: utf-8 -*-

"""
cherry.tokenizer
~~~~~~~~~~~~
This module implements text tokenizer.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
from .config import DATA_DIR, LAN_DICT
from .exceptions import LanguageNotFoundError


class Token:
    '''
    Tokenizer text implements
    '''
    def __init__(self, **kwargs):
        self.lan = kwargs['lan']
        # Get stop word in lan directory
        self.stop_word = self._get_stop_word()
        split_fun = LAN_DICT[self.lan]['split']
        self.tokenizer = split_fun(kwargs['text'], self.stop_word)

    def _get_stop_word(self):
        '''
        Stop word should store in the stop_word.dat under language directory.
        '''
        try:
            lan_data_path = os.path.join(
                DATA_DIR, ('data/' + self.lan + '/'))
            stop_word_path = os.path.join(
                lan_data_path, 'stop_word.dat')
            with open(stop_word_path, encoding='utf-8') as f:
                stop_word = [l[:-1] for l in f.readlines()]
        except IOError:
            error = (
                'Language {0} not found'.format(self.lan))
            raise LanguageNotFoundError(error)
        return stop_word
