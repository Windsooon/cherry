import os
import shutil
import unittest
import codecs
import pickle
from unittest import mock
import tempfile
import cherry
from cherry.datasets import BUILD_IN_MODELS
from cherry.base import *
from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer, HashingVectorizer

class UseModel:

    def __init__(self, model, cache=True, cache_problem=False):
        self.dir_path = os.path.join(DATA_DIR, model)
        self.cache = cache
        self.cache_problem = cache_problem

    def __enter__(self):
        os.mkdir(self.dir_path)
        if self.cache:
            # Create cache files with wrong format
            if self.cache_problem:
                with open(os.path.join(self.dir_path, 'foo.pkz'), 'wb') as f:
                    f.write(b'Wrong Format')
            else:
                data = {'all': {'data': 'bar'}}
                compressed_content = codecs.encode(pickle.dumps(data), 'zlib_codec')
                with open(os.path.join(self.dir_path, 'foo.pkz'), 'wb') as f:
                    f.write(compressed_content)
            return f

    def __exit__(self, *args):
        shutil.rmtree(self.dir_path)

class BaseTest(unittest.TestCase):

    def setUp(self):
        self.foo_model = 'foo'
        self.news_model = 'newsgroups'
        self.foo_model_path = os.path.join(DATA_DIR, self.foo_model)
        self.news_model_path = os.path.join(DATA_DIR, self.news_model)

    # get_stop_words()
    def test_stop_words(self):
        self.assertIn('anyone', get_stop_words())
        self.assertIn('的', get_stop_words(language='Chinese'))
        self.assertNotIn('你好', get_stop_words(language='Chinese'))
        self.assertNotIn('human', get_stop_words())

    # load_data()
    @mock.patch('cherry.base._load_data_from_local')
    def test_load_data_found(self, mock_load_files):
        with UseModel(self.foo_model) as model:
            load_data(self.foo_model)
        mock_load_files.assert_called_once_with(
            self.foo_model, categories=None, encoding=None)

    @mock.patch('cherry.base._load_data_from_local')
    def test_load_data_not_found(self, mock_load_files):
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            load_data(self.foo_model)
        self.assertEqual(
            str(filesNotFoundError.exception),
            'foo is not built in models and not found in dataset folder.')

    # _load_data_from_local()
    @mock.patch('cherry.base.load_files')
    def test_load_local_data_from_local_without_cache(self, mock_load_files):
        mock_load_files.return_value = None
        with UseModel(self.foo_model, cache=False) as model:
            res = cherry.base._load_data_from_local(self.foo_model)
            self.assertEqual(res, None)
            mock_load_files.assert_called_once_with(self.foo_model_path, categories=None, encoding=None)
            # Create new cache files
            cache_path = os.path.join(self.foo_model_path, self.foo_model + '.pkz')
            self.assertTrue(os.path.exists(cache_path))

    def test_load_local_data_from_local_with_cache(self):
        with UseModel(self.foo_model) as model:
            res = cherry.base._load_data_from_local(self.foo_model)
        self.assertEqual(res['data'], 'bar')

    def test_load_local_data_from_local_with_cache_failed(self):
        with UseModel(self.foo_model, cache_problem=True) as model:
            with self.assertRaises(cherry.exceptions.NotSupportError) as notFoundError:
                res = cherry.base._load_data_from_local(self.foo_model)
            self.assertEqual(
                str(notFoundError.exception),
                'Can\'t load cached data from foo. Please try again after delete those cache files.')

    # _load_data_from_remote()
    @mock.patch('cherry.base._load_data_from_remote')
    def test_load_data_from_remote(self, mock_load_files):
        load_data(self.foo_model)
        mock_load_files.assert_called_once_with(
            self.foo_model, categories=None, encoding=None)

    def test_load_data_from_remote_not_build_in(self):
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            cherry.base._load_data_from_remote(self.foo_model)
        self.assertEqual(
            str(filesNotFoundError.exception),
            'foo is not built in models and not found in dataset folder.')

    @mock.patch('cherry.base._load_data_from_local')
    @mock.patch('cherry.base._decompress_data')
    @mock.patch('cherry.base._fetch_remote')
    def test_load_data_from_remote_download(self, mock_fetch_remote, mock_decompress_data, mock_load_data_from_local):
        model_existed = os.path.exists(self.news_model_path)
        info = BUILD_IN_MODELS[self.news_model]
        cherry.base._load_data_from_remote(self.news_model)
        self.assertTrue(os.path.exists(self.news_model_path) is True)
        if not model_existed:
            shutil.rmtree(self.news_model_path)
        mock_load_data_from_local.assert_called_once_with(
            self.news_model, categories=None, encoding=info[3])

    # get_tokenizer()
    def test_get_tokenizer_function(self):
        self.assertEqual(get_tokenizer('English').__name__, 'word_tokenize')
        self.assertEqual(get_tokenizer('Chinese').__name__, 'cut')
        with self.assertRaises(cherry.exceptions.NotSupportError) as notFoundError:
            self.assertEqual(get_tokenizer('Foo').__name__, 'cut')
        self.assertEqual(
            str(notFoundError.exception),
            'You need to specify tokenizer function when the language is nor English or Chinese.')

    # get_vectorizer_and_clf()
    @mock.patch('cherry.base.get_clf')
    @mock.patch('cherry.base.get_vectorizer')
    def test_get_vectorizer_and_clf_default(self, mock_get_vectorizer, mock_get_clf):
        mock_get_vectorizer.return_value = 'Count'
        mock_get_clf.return_value = 'MNB'
        candidates = [('English', 'Count', 'MNB'), ('Chinese', 'Tfidf', 'Random')]
        for candidate in candidates:
            language, vectorizer_method, clf_method = candidate
            res = get_vectorizer_and_clf(
                language, None, None, vectorizer_method, clf_method)
            mock_get_vectorizer.assert_called_with(language, vectorizer_method)
            mock_get_clf.assert_called_with(clf_method)
            self.assertEqual(res, ('Count', 'MNB'))

    @mock.patch('cherry.base.get_clf')
    @mock.patch('cherry.base.get_vectorizer')
    def test_get_vectorizer_and_clf_custom(self, mock_get_vectorizer, mock_get_clf):
        candidates = ('English', 'Random', 'String', 'Count', 'MNB')
        # language, vectorizer, clf, vectorizer_method, clf_method = candidate
        res = get_vectorizer_and_clf(*candidates)
        mock_get_vectorizer.assert_not_called()
        mock_get_clf.assert_not_called()
        self.assertEqual(res, ('Random', 'String'))

    # get_vectorizer()
    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.base.get_tokenizer')
    def test_get_vectorizer_default(self, mock_get_tokenizer, mock_get_stop_words):
        language_lst = ['English', 'Chinese']
        for language in language_lst:
            res = get_vectorizer(language, 'Count')
            mock_get_tokenizer.assert_called_with(language)
            mock_get_stop_words.assert_called_with(language)
            self.assertEqual(type(res), CountVectorizer)

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.base.get_tokenizer')
    def test_get_vectorizer_tfidf(self, mock_get_tokenizer, mock_get_stop_words):
        language_lst = ['English', 'Chinese']
        for language in language_lst:
            res = get_vectorizer(language, 'Tfidf')
            mock_get_tokenizer.assert_called_with(language)
            mock_get_stop_words.assert_called_with(language)
            self.assertEqual(type(res), TfidfVectorizer)

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.base.get_tokenizer')
    def test_get_vectorizer_failed(self, mock_get_tokenizer, mock_get_stop_words):
        language = 'English'
        with self.assertRaises(cherry.exceptions.MethodNotFoundError) as methodNotFoundError:
            res = get_vectorizer(language, 'Foo')
        self.assertEqual(
            str(methodNotFoundError.exception),
            'Please make sure vectorizer_method in "Count", "Tfidf" or "Hashing".')

    # get_clf()
    def test_get_clf_failed(self):
        with self.assertRaises(cherry.exceptions.MethodNotFoundError) as methodNotFoundError:
            res = get_clf('Foo')
        self.assertEqual(
            str(methodNotFoundError.exception),
            'Please make sure clf_method in "MNB", "SGD", "RandomForest" or "AdaBoost".')
