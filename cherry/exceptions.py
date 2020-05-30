# -*- coding: utf-8 -*-
"""
cherry.exceptions
~~~~~~~~~~~~~~~~~~~
This module contains the set of cheery' exceptions.
:copyright: (c) 2018-2020 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

__all__ = ['NotSupportError',
           'CacheNotFoundError',
           'FilesNotFoundError',
           'DownloadError',
           'MethodNotFoundError',
           'TokenNotFoundError',
           'DataMismatchError',
           'UnicodeFileEncodeError']

class CherryException(Exception):
    '''Base Exception'''
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return self.error

class NotSupportError(CherryException):
    '''Not support this feature'''

class CacheNotFoundError(CherryException):
    '''Cache files not found'''

class FilesNotFoundError(CherryException):
    '''Files not found'''

class DownloadError(CherryException):
    '''Download data error'''

class MethodNotFoundError(CherryException):
    '''Method not found'''

class TokenNotFoundError(CherryException):
    '''Token not found in training data'''

class DataMismatchError(CherryException):
    '''Data mismatch'''

class UnicodeFileEncodeError(CherryException):
    '''Unicode File Encode Error'''
