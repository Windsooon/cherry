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

class FilesNotFoundError(IOError):
    '''Files not found'''

class MethodNotFoundError(AttributeError):
    '''Method not found'''

class DataMismatchError(AttributeError):
    '''Data mismatch'''

class UnicodeFileEncodeError(UnicodeEncodeError):
    '''Unicode File Encode Error'''
