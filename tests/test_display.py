import os
import unittest
from unittest import mock
from cherry.displayer import Display
from cherry.base import DATA_DIR, load_data

class DisplayTest(unittest.TestCase):

    def setUp(self):
        self.x_data = [1, 2]
        self.y_data = [0, 1]

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.displayer.load_data')
    @mock.patch('cherry.displayer.Display.display_learning_curve')
    def test_init_load_data(self, mock_learning, mock_load, mock_stop_words):
        mock_stop_words.return_value = ['first', 'second']
        mock_load.return_value = self.x_data, self.y_data
        d = Display(model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=None, y_data=None)
        mock_load.assert_called_once_with('harmful')

    @mock.patch('cherry.displayer.load_data')
    @mock.patch('cherry.displayer.get_vectorizer')
    @mock.patch('cherry.displayer.Display.display_learning_curve')
    def test_init_no_vect(self, mock_learning, mock_vect, mock_load):
        mock_load.return_value = self.x_data, self.y_data
        d = Display(model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=None, y_data=None)
        mock_vect.assert_called_once_with('harmful', None)

    @mock.patch('cherry.displayer.load_data')
    @mock.patch('cherry.displayer.get_vectorizer')
    @mock.patch('cherry.displayer.get_clf')
    @mock.patch('cherry.displayer.Display.display_learning_curve')
    def test_init_no_clf(self, mock_learning, mock_clf, mock_vect, mock_load):
        mock_load.return_value = self.x_data, self.y_data
        d = Display(model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=None, y_data=None)
        mock_clf.assert_called_once_with('harmful', None)


    @mock.patch('cherry.displayer.load_data')
    @mock.patch('cherry.displayer.get_vectorizer')
    @mock.patch('cherry.displayer.Display.plot_learning_curve')
    def test_init_no_clf_vect(self, mock_learning, mock_vect, mock_load):
        mock_load.return_value = self.x_data, self.y_data
        d = Display(model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=None, y_data=None)
        mock_learning.assert_called_once()
