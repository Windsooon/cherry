# -*- coding: utf-8 -*-
"""
cherry.exceptions
~~~~~~~~~~~~~~~~~~~
This module contains the set of cheery' exceptions.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""


class CacheNotFoundError(IOError):
    '''Cache files not found'''

class StopWordsNotFoundError(IOError):
    '''Stop words files not found'''

class MethodNotFoundError(AttributeError):
    '''method not found'''

class UnicodeFileEncodeError(UnicodeEncodeError):
    '''Unicode File Encode Error'''
