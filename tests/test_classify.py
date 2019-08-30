import os
import unittest
from unittest import mock
import cherry


class ClassifyTest(unittest.TestCase):

    def setUp(self):
        pass

    @mock.patch('cherry.classifyer.Classify._classify')
    @mock.patch('cherry.classifyer.load_cache')
    def test_load_cache(self, mock_load, mock_classify):
        mock_load.return_value = 'random'
        mock_classify.return_value = [1, 0], ['random', 'text']
        c = cherry.classifyer.Classify(text=['random text'], N=20)
        self.assertEqual(c.probability, [1, 0])
        self.assertEqual(c.word_list, ['random', 'text'])
