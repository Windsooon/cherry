import os
import cherry
import unittest

from unittest import mock
from cherry.base import DATA_DIR, load_data

class DisplayTest(unittest.TestCase):

    # __init__()
    @mock.patch('cherry.displayer.Display.display_learning_curve')
    @mock.patch('cherry.displayer.load_all')
    def test_init(self, mock_load, mock_display):
        mock_load.return_value = ([1], [0], 'vectorizer', 'clf')
        cherry.displayer.Display(model='foo')
        mock_load.assert_called_with(
            'foo', categories=None, clf=None, clf_method=None,
            encoding=None, language=None, preprocessing=None,
            vectorizer=None, vectorizer_method=None, x_data=None, y_data=None)
        mock_display.assert_called_with('vectorizer', 'clf', [1], [0])
