# -*- coding: utf-8 -*-

"""
cherry.exceptions
~~~~~~~~~~~~~~~~~~~

This module contains the set of cheery' exceptions.
"""


class CherryIOException(IOError):

    def __init__(self, *args, **kwargs):
        super(CherryIOException, self).__init__(*args, **kwargs)

class CacheNotFoundError(CherryIOException, IOError):
    '''Cache files not found'''

class StopWordsNotFoundError(CherryIOException, IOError):
    '''Stop words files not found'''
