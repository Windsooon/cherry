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

    # __init__()
    def test_cache_not_found(self):
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            t = Trainer(model='foo')

    @mock.patch('cherry.trainer.write_cache')
    @mock.patch('cherry.trainer.Trainer._train')
    @mock.patch('cherry.trainer.Trainer._get_vectorizer_and_clf')
    @mock.patch('cherry.trainer.load_data')
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
            mock_load_data.assert_called_with(self.foo_model, categories=None, encoding=None)
            mock_get.assert_called_with(
                language, {'vectorizer_method': vectorizer_method, 'clf_method': clf_method})
            mock_train.assert_called_with('vectorizer', 'clf', meta_data_c(data=['random'], target=[2]))
            mock_write_cache.assert_called_with('foo', 'clf', 'clf.pkz')

    # _get_vectorizer_and_clf()
    @mock.patch('cherry.trainer.get_clf')
    @mock.patch('cherry.trainer.get_vectorizer')
    def test_get_vectorizer_and_clf_default(self, mock_get_vectorizer, mock_get_clf):
        t = Trainer._get_vectorizer_and_clf('English', {'vectorizer_method': 'Count', 'clf_method': 'MNB'})
        mock_get_vectorizer.assert_called_with('English', 'Count')
        mock_get_clf.assert_called_with('MNB')

    @mock.patch('cherry.trainer.get_clf')
    @mock.patch('cherry.trainer.get_vectorizer')
    def test_get_vectorizer_and_clf_custom(self, mock_get_vectorizer, mock_get_clf):
        data = {'vectorizer': 'random', 'clf': 'string', 'vectorizer_method': 'Count', 'clf_method': 'MNB'}
        t = Trainer._get_vectorizer_and_clf('English', data)
        mock_get_vectorizer.assert_not_called()
        mock_get_clf.assert_not_called()
        self.assertEqual(t, ('random', 'string'))

    # _train()
    @mock.patch('cherry.trainer.Pipeline.fit')
    def test_train_default(self, mock_fix):
        meta_data_c = namedtuple('meta_data_c', ['data', 'target'])
        mock_data = meta_data_c(data=['random'], target=[2])
        t = Trainer._train(CountVectorizer, MultinomialNB, mock_data)
        mock_fix.assert_called_with(['random'], [2])
