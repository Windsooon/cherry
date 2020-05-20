import os
import shutil
import unittest
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
        with open(os.path.join(self.dir_path, 'foo.pkz'), 'wb+') as f:
            file = f.write(b'bar')
        return file

    def __exit__(self, *args):
        shutil.rmtree(self.dir_path)

class BaseTest(unittest.TestCase):

    def test_stop_words(self):
        self.assertIn('anyone', get_stop_words())
        self.assertNotIn('human', get_stop_words())

    def test_load_data_from_remote_not_build_in(self):
        model_path = os.path.join(DATA_DIR, 'foo')
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            _load_data_from_remote(model_path, 'foo')
        self.assertEqual(
            str(filesNotFoundError.exception),
            'foo is not built in models and not found in dataset folder.')


    @mock.patch('cherry.base._load_data_from_local')
    @mock.patch('cherry.base._download_data')
    def test_load_data_from_remote_download(self, mock_download_data, mock_from_local):
        model = 'newsgroups'
        model_path = os.path.join(DATA_DIR, model)
        res = _load_data_from_remote(os.path.join(DATA_DIR, model), model)
        mock_from_local.return_value = 'foo'
        mock_download_data.assert_called_once_with(
            BUILD_IN_MODELS[model], model_path, None, None)

    @mock.patch('cherry.base._load_data_from_remote')
    def test_load_data_from_remote(self, mock_load_files):
        load_data('foo')
        model_path = os.path.join(DATA_DIR, 'foo')
        mock_load_files.assert_called_once_with(
            model_path, 'foo', categories=None, encoding=None)

    @mock.patch('cherry.base._load_data_from_local')
    def test_load_data_found(self, mock_load_files):
        with UseModel('foo') as model:
            load_data('foo')
        mock_load_files.assert_called_once_with('/Users/windson/learn/cherry/cherry/datasets/foo', 'foo', categories=None, encoding=None)

    @mock.patch('cherry.base.load_files')
    def test_load_local_data_from_local_without_cache(self, mock_load_files):
        mock_load_files.return_value = None
        res = _load_data_from_local('foo', 'foo')
        self.assertEqual(res, None)
        mock_load_files.assert_called_once_with('foo', categories=None, encoding=None)

    def test_load_local_data_from_local_with_cache(self):
        with UseModel('foo') as model:
            res = _load_data_from_local(os.path.join(DATA_DIR, 'foo'), 'foo')
        self.assertEqual(res['data'], [])

    @mock.patch('cherry.base.pickle.loads')
    def test_load_local_data_from_local_with_cache_failed(self, load_data):
        load_data.side_effect = cherry.exceptions.NotSupportError('not found')
        # self._create_model('foo')
        # res = _load_data_from_local(os.path.join(DATA_DIR, 'foo'), 'foo')
        # self._delete_model('foo')
        # self.assertEqual(res['data'], [])

    @mock.patch('cherry.base.urlretrieve')
    def test_download_data(self, url_data):
        url_data.side_effect = cherry.exceptions.FilesNotFoundError('not found')
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            _download_data(('abc.com', 'foo', 'abc'), '/User/example', None, None)

