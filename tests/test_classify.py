import os
import unittest
from unittest import mock
import cherry
from sklearn.exceptions import NotFittedError


class ClassifyTest(unittest.TestCase):

    def setUp(self):
        pass

    @mock.patch('cherry.classifyer.Classify._classify')
    @mock.patch('cherry.classifyer.Classify._load_cache')
    def test_load_cache(self, mock_load, mock_classify):
        mock_classify.return_value = [1, 0], ['random', 'text']
        c = cherry.classifyer.Classify(model='harmful', text=['random text'], N=20)
        mock_load.assert_called_once_with('harmful')

    @mock.patch('cherry.classifyer.Classify._classify')
    @mock.patch('cherry.classifyer.load_cache')
    def test_load_base_cache(self, mock_load, mock_classify):
        mock_classify.return_value = [1, 0], ['random', 'text']
        c = cherry.classifyer.Classify(model='harmful', text=['random text'], N=20)
        mock_load.assert_called_with('harmful', 've.pkl')

    @mock.patch('cherry.classifyer.Classify._classify')
    @mock.patch('cherry.classifyer.load_cache')
    def test_classify(self, mock_load, mock_classify):
        mock_load.return_value = 'random'
        mock_classify.return_value = [1, 0], ['random', 'text']
        c = cherry.classifyer.Classify(model='harmful', text=['random text'], N=20)
        self.assertEqual(c.probability, [1, 0])
        self.assertEqual(c.word_list, ['random', 'text'])

    @mock.patch('sklearn.feature_extraction.text.CountVectorizer.transform')
    @mock.patch('cherry.classifyer.load_cache')
    def test_classify_with_missing_token(self, mock_load, mock_trans):
        mock_object = mock.Mock()
        mock_object.transform.side_effect = NotFittedError()
        mock_load.return_value = mock_object
        with self.assertRaises(cherry.exceptions.TokenNotFoundError) as token_error:
            c = cherry.classifyer.Classify(model='harmful', text=['random text'], N=20)
        self.assertEqual(
            str(token_error.exception),
            'Some of the tokens in text never appear in training data')
