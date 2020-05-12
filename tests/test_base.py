import unittest
from unittest import mock
import cherry
from cherry.base import *


class BaseTest(unittest.TestCase):

    def test_stop_words(self):
        self.assertIn("anyone", get_stop_words())

    def test_load_data_not_found(self):
        with self.assertRaises(cherry.exceptions.FilesNotFoundError) as filesNotFoundError:
            load_data("harmful")
        self.assertEqual(
            str(filesNotFoundError.exception),
            'harmful is not built in models and not found in dataset folder.')
