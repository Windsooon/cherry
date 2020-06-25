import unittest
from unittest import mock
import cherry


class ApiTest(unittest.TestCase):

    def setUp(self):
        self.model = 'foo'
        self.text = 'random string'

    @mock.patch('cherry.api.Classify')
    def test_classify_api(self, mock_classify):
        cherry.classify(model=self.model, text=self.text)
        mock_classify.assert_called_once_with(model=self.model, text=self.text)

    @mock.patch('cherry.api.Trainer')
    def test_train_api(self, mock_train):
        cherry.train(model=self.model)
        mock_train.assert_called_once_with(
            self.model, categories=None, clf=None, clf_method='MNB',
            encoding='utf-8', language='English', preprocessing=None, vectorizer=None,
            vectorizer_method='Count', x_data=None, y_data=None)

    @mock.patch('cherry.api.Trainer')
    def test_api_call_model_clf_vectorizer(self, mock_trainer):
        cherry.train('foo', clf='clf', vectorizer='vectorizer')
        mock_trainer.assert_called_with(
            'foo', preprocessing=None, categories=None, encoding='utf-8', clf='clf', clf_method='MNB', language='English',
            vectorizer='vectorizer', vectorizer_method='Count', x_data=None, y_data=None)

    @mock.patch('cherry.api.Performance')
    def test_performance_api(self, mock_performance):
        cherry.performance(model=self.model)
        mock_performance.assert_called_once_with(
            self.model, categories=None, clf=None, clf_method='MNB', encoding='utf-8',
            language='English', n_splits=10, output='Stdout', preprocessing=None,
            vectorizer=None, vectorizer_method='Count', x_data=None, y_data=None)

    @mock.patch('cherry.api.Performance')
    def test_performance_api_model_clf_vectorizer(self, mock_performance):
        cherry.performance('foo', clf='clf', vectorizer='vectorizer')
        mock_performance.assert_called_with(
            'foo', categories=None, clf='clf', clf_method='MNB',
            encoding='utf-8', language='English', n_splits=10,
            output='Stdout', preprocessing=None, vectorizer='vectorizer',
            vectorizer_method='Count', x_data=None, y_data=None)

    @mock.patch('cherry.api.Search')
    def test_api_call(self, mock_search):
        cherry.search('foo', {'foo': 'bar'})
        mock_search.assert_called_with(
            'foo', {'foo': 'bar'}, categories=None, clf=None, clf_method='MNB', cv=3,
            encoding='utf-8', language='English', method='RandomizedSearchCV', n_jobs=-1,
            preprocessing=None, vectorizer=None, vectorizer_method='Count', x_data=None, y_data=None)

    @mock.patch('cherry.api.Display')
    def test_display_api(self, mock_display):
        cherry.display(model=self.model)
        mock_display.assert_called_once_with(
            self.model, categories=None, clf=None, clf_method='MNB',
            encoding='utf-8', language='English', preprocessing=None,
            vectorizer=None, vectorizer_method='Count', x_data=None, y_data=None)
