import os
import unittest
import cherry
from unittest import mock
from cherry.base import DATA_DIR, load_data

class PerformanceTest(unittest.TestCase):

    def setUp(self):
        pass

    # api call
    @mock.patch('cherry.api.Performance')
    def test_api_call_only_model(self, mock_performance):
        cherry.performance('foo')
        mock_performance.assert_called_with(
            'foo', categories=None, clf=None, clf_method='MNB',
            encoding=None, language='English', n_splits=10,
            output='Stdout', preprocessing=None, vectorizer=None,
            vectorizer_method='Count', x_data=None, y_data=None)
