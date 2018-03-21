# -*- coding: utf-8 -*-

"""
cherry.tokenizer
~~~~~~~~~~~~
This module implements text tokenizer.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import os
import jieba
from .config import DATA_DIR
from .exceptions import LanguageNotFoundError


class Token:
    '''
    Tokenizer text implements
    '''
    def __init__(self, **kwargs):
        self.lan = kwargs['lan']
        # Get stop word in lan directory
        self.stop_word = self._get_stop_word()
        if kwargs['split']:
            self.tokenizer = list(
                kwargs['split'](kwargs['text']))
        else:
            self.tokenizer = self._get_tokenizer(kwargs['text'], kwargs['lan'])

    def _get_tokenizer(self, text, lan):
        if lan == 'Chinese':
            return [
                t for t in jieba.cut(text) if len(t) > 1
                and t not in self.stop_word]

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
