import os
import unittest
import cherry

from collections import namedtuple
from unittest import mock
from cherry.trainer import Trainer
from cherry.base import DATA_DIR, load_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.foo_model = 'foo'
        self.news_model = 'newsgroups'
        self.foo_model_path = os.path.join(DATA_DIR, self.foo_model)
        self.news_model_path = os.path.join(DATA_DIR, self.news_model)

    @mock.patch('cherry.api.Trainer')
    def test_api_call_model_clf_vectorizer(self, mock_trainer):
        cherry.train('foo', clf='clf', vectorizer='vectorizer')
        mock_trainer.assert_called_with(
            'foo', preprocessing=None, categories=None, encoding=None, clf='clf', clf_method='MNB', language='English',
            vectorizer='vectorizer', vectorizer_method='Count', x_data=None, y_data=None)

    # __init__()
    def test_cache_not_found(self):
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            t = Trainer(model='foo')

    @mock.patch('cherry.trainer.write_cache')
    @mock.patch('cherry.trainer.Trainer._train')
    @mock.patch('cherry.base.get_vectorizer_and_clf')
    @mock.patch('cherry.base.load_data')
    def test_mock_init_call(self, mock_load_data, mock_get, mock_train, mock_write_cache):
        meta_data_c = namedtuple('meta_data_c', ['data', 'target'])
        mock_load_data.return_value = meta_data_c(data=['random'], target=[2])
        mock_get.return_value = ('vectorizer', 'clf')
        candidates = [('English', 'Count', 'MNB'), ('Chinese', 'Tfidf', 'Random')]
        for candidate in candidates:
            language, vectorizer_method, clf_method = candidate
            t = Trainer(
                model=self.foo_model, language=language,
                vectorizer_method=vectorizer_method, clf_method=clf_method)
            mock_load_data.assert_called_with(
                self.foo_model, categories=None, encoding=None)
            mock_get.assert_called_with(
                language, None, None, vectorizer_method, clf_method)
            mock_train.assert_called_with(
                'vectorizer', 'clf', ['random'], [2])
            mock_write_cache.assert_called_with('foo', 'clf', 'clf.pkz')

    # _train()
    @mock.patch('cherry.trainer.Pipeline.fit')
    def test_train_default(self, mock_fix):
        x_data, y_data = ['random'], [2]
        t = Trainer._train(CountVectorizer, MultinomialNB, x_data, y_data)
        mock_fix.assert_called_with(['random'], [2])
