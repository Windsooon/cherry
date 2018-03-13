# -*- coding: utf-8 -*-

"""
cherry.exceptions
~~~~~~~~~~~~~~~~~~~

This module contains the set of cheery' exceptions.
"""


class CherryIOException(IOError):

    def __init__(self, *args, **kwargs):
        super(CherryIOException, self).__init__(*args, **kwargs)


class LanguageNotFoundError(CherryIOException, ValueError):
    '''An language not found error occurred.'''


class TestDataNumError(CherryIOException, IndexError):
    '''Test data number should between 0 to all data'''


class CacheNotFoundError(CherryIOException, IOError):
    '''Cache files not found'''
