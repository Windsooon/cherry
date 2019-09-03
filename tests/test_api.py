import unittest
from unittest import mock
import cherry


class ApiTest(unittest.TestCase):

    @mock.patch('cherry.api.Classify')
    def test_classify_api(self, mock_classify):
        cherry.classify(['random string'], model='harmful')
        mock_classify.assert_called_once_with(
            N=20, model='harmful', text=['random string'])

    @mock.patch('cherry.api.Trainer')
    def test_train_api(self, mock_train):
        cherry.train(model='harmful')
        mock_train.assert_called_once_with(
            'harmful', clf=None, clf_method=None, vectorizer=None,
            vectorizer_method=None, x_data=None, y_data=None)

    @mock.patch('cherry.api.Performance')
    def test_performance_api(self, mock_performance):
        cherry.performance(model='harmful')
        mock_performance.assert_called_once_with(
            'harmful', clf=None, clf_method=None, n_splits=5, output='Stdout',
            vectorizer=None, vectorizer_method=None, x_data=None, y_data=None)

    @mock.patch('cherry.api.Search')
    def test_search_api(self, mock_search):
        cherry.search(model='harmful', parameters={})
        mock_search.assert_called_once_with(
            'harmful', clf=None, clf_method=None, cv=3, iid=False, method='RandomizedSearchCV',
            n_jobs=1, parameters={}, vectorizer=None, vectorizer_method=None, x_data=None, y_data=None)

    @mock.patch('cherry.api.Display')
    def test_display_api(self, mock_display):
        cherry.display(model='harmful')
        mock_display.assert_called_once_with(
            'harmful', clf=None, clf_method=None, vectorizer=None,
            vectorizer_method=None, x_data=None, y_data=None)

