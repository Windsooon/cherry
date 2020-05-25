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

class UseModel():

    def __init__(self, model):
        self.dir_path = os.path.join(DATA_DIR, model)

    def __enter__(self):
        os.mkdir(self.dir_path)
        data = {'data': 'bar'}
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
        mock_load_files.assert_called_once_with(self.foo_model, categories=None, encoding=None)

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
        res = _load_data_from_local(self.foo_model)
        self.assertEqual(res, None)
        mock_load_files.assert_called_once_with(self.foo_model_path, categories=None, encoding=None)

    def test_load_local_data_from_local_with_cache(self):
        with UseModel(self.foo_model) as model:
            res = _load_data_from_local(self.foo_model)
        self.assertEqual(res['data'], 'bar')

    def test_load_local_data_from_local_with_cache_failed(self):
        with UseModel(self.foo_model) as model:
            res = _load_data_from_local(self.foo_model_path)
        self.assertEqual(res['data'], [])

    # _load_data_from_remote()
    @mock.patch('cherry.base._load_data_from_remote')
    def test_load_data_from_remote(self, mock_load_files):
        load_data(self.foo_model)
        mock_load_files.assert_called_once_with(
            self.foo_model, categories=None, encoding=None)

    def test_load_data_from_remote_not_build_in(self):
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            _load_data_from_remote(self.foo_model)
        self.assertEqual(
            str(filesNotFoundError.exception),
            'foo is not built in models and not found in dataset folder.')

    @mock.patch('cherry.base._load_data_from_local')
    @mock.patch('cherry.base._decompress_data')
    @mock.patch('cherry.base._fetch_remote')
    def test_load_data_from_remote_download(self, mock_fetch_remote, mock_decompress_data, mock_load_data_from_local):
        self.assertTrue(os.path.exists(self.news_model_path) is False)
        _load_data_from_remote(self.news_model)
        self.assertTrue(os.path.exists(self.news_model_path) is True)
        shutil.rmtree(self.news_model_path)
        mock_load_data_from_local.assert_called_once_with(
            self.news_model, categories=None, encoding=None)
