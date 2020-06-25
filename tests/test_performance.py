import os
import unittest
import cherry
from unittest import mock
from cherry.base import DATA_DIR, load_data

class PerformanceTest(unittest.TestCase):

    # __init__()
    @mock.patch('cherry.performancer.load_all')
    def test_init(self, mock_load):
        mock_load.return_value = ([1], [0], 'vectorizer', 'clf')
        cherry.performancer.Performance('foo')
        mock_load.assert_called_with(
            'foo', categories=None, clf=None, clf_method=None,
            encoding=None, language=None, preprocessing=None,
            vectorizer=None, vectorizer_method=None, x_data=None, y_data=None)
