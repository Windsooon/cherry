import os
import unittest
import cherry

from unittest import mock
from cherry import classify
from sklearn.exceptions import NotFittedError


class ClassifyTest(unittest.TestCase):

    def setUp(self):
        pass

    # __init__()

    @mock.patch('cherry.classifyer.Classify._classify')
    @mock.patch('cherry.classifyer.Classify._load_cache')
    def test_init(self, mock_load, mock_classify):
        mock_load.return_value = ('foo', 'bar')
        res = cherry.classifyer.Classify(model='random', text=['random text'])
        if res.get_CACHE() == False:
            mock_load.assert_called_once_with('random')
        mock_classify.assert_called_once_with(['random text'])

    # _load_cache()

    @mock.patch('cherry.classifyer.Classify._classify')
    @mock.patch('cherry.classifyer.load_cache')
    def test_load_cache(self, mock_load, mock_classify):
        res = cherry.classifyer.Classify(model='foo', text=['random text'])
        mock_load.assert_not_called()
