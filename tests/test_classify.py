import os
import unittest
from unittest import mock
import cherry
from sklearn.exceptions import NotFittedError


class ClassifyTest(unittest.TestCase):

    def setUp(self):
        pass

    # api call
    @mock.patch('cherry.api.Classify')
    def test_api_call_model_text(self, mock_classify):
        cherry.classify('foo', 'random text')
        mock_classify.assert_called_with(model='foo', text='random text')

    # __init__()
    @mock.patch('cherry.classifyer.Classify._classify')
    @mock.patch('cherry.classifyer.Classify._load_cache')
    def test_init(self, mock_load, mock_classify):
        mock_classify.return_value = [1, 0], ['random', 'text']
        res = cherry.classifyer.Classify(model='harmful', text=['random text'])
        mock_load.assert_called_once_with('harmful')
        mock_classify.assert_called_once_with(['random text'])

    # _load_cache()
    @mock.patch('cherry.classifyer.Classify._classify')
    @mock.patch('cherry.classifyer.load_cache')
    def test_load_cache(self, mock_load, mock_classify):
        mock_classify.return_value = [1, 0], ['random', 'text']
        res = cherry.classifyer.Classify(model='harmful', text=['random text'])
        mock_load.assert_called_with('harmful', 've.pkz')


    @mock.patch('sklearn.feature_extraction.text.CountVectorizer.transform')
    @mock.patch('cherry.classifyer.load_cache')
    def test_classify_with_missing_token(self, mock_load, mock_trans):
        mock_object = mock.Mock()
        mock_object.transform.side_effect = NotFittedError()
        mock_load.return_value = mock_object
        with self.assertRaises(cherry.exceptions.TokenNotFoundError) as token_error:
            res = cherry.classifyer.Classify(model='harmful', text=['random text'])
        self.assertEqual(
            str(token_error.exception),
            'Some of the tokens in text never appear in training data')
