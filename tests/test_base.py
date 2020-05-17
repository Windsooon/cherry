import os
import shutil
import unittest
from unittest import mock
import tempfile
import cherry
from cherry.base import *


class BaseTest(unittest.TestCase):

    def test_stop_words(self):
        self.assertIn("anyone", get_stop_words())
        self.assertNotIn("human", get_stop_words())

    def test_load_data_not_found(self):
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            load_data("harmful")
        self.assertEqual(
            str(filesNotFoundError.exception),
            'harmful is not built in models and not found in dataset folder.')

    @mock.patch('cherry.base.load_files')
    def test_load_local_data_from_local_without_cache(self, mock_load_files):
        mock_load_files.return_value = None
        res = _load_data_from_local('foo', 'foo')
        self.assertEqual(res, None)
        mock_load_files.assert_called_once_with('foo', categories=None, encoding=None)

    def test_load_local_data_from_local_with_cache(self):
        dir_path = os.path.join(DATA_DIR, 'foo')
        os.mkdir(dir_path)
        with open(os.path.join(dir_path, 'foo.pkz'),'wb+') as f:
            f.write(b'bar')
        res = _load_data_from_local(os.path.join(DATA_DIR, 'foo'), 'foo')
        self.assertEqual(res['data'], [])
        shutil.rmtree(dir_path)

    def test_load_data_from_remote_not_build_in(self):
        pass

