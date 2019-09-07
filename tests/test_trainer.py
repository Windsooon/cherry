import os
import unittest
from unittest import mock
from cherry.trainer import Trainer
from cherry.base import DATA_DIR, load_data

class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.x_data = [1, 2]
        self.y_data = [0, 1]

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.trainer.Trainer.train')
    def test_write_cache(self, mock_train, mock_stop_words):
        '''
        TODO: use create special cache files for testing
        '''
        mock_stop_words.return_value = ['first', 'second']
        t = Trainer(
            model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=self.x_data, y_data=self.y_data)
        self.assertTrue(os.path.exists(os.path.join(DATA_DIR, 'harmful/trained.pkl')))
        self.assertTrue(os.path.exists(os.path.join(DATA_DIR, 'harmful/ve.pkl')))

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.trainer.load_data')
    @mock.patch('cherry.trainer.Trainer._write_cache')
    @mock.patch('cherry.trainer.Trainer.train')
    def test_init_load_data(self, mock_train, mock_cache, mock_load, mock_stop_words):
        mock_stop_words.return_value = ['first', 'second']
        mock_load.return_value = self.x_data, self.y_data
        t = Trainer(
            model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=None, y_data=None)
        mock_load.assert_called_once_with('harmful')

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.trainer.load_data')
    @mock.patch('cherry.trainer.Trainer._write_cache')
    @mock.patch('cherry.trainer.Trainer.train')
    def test_init_pass_data(self, mock_train, mock_cache, mock_load, mock_stop_words):
        mock_stop_words.return_value = ['first', 'second']
        mock_load.return_value = self.x_data, self.y_data
        t = Trainer(
            model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=self.x_data, y_data=self.y_data)
        mock_load.assert_not_called()

    @mock.patch('cherry.trainer.get_vectorizer')
    @mock.patch('cherry.trainer.Trainer._write_cache')
    @mock.patch('cherry.trainer.Trainer.train')
    def test_init_no_vectorizer(self, mock_train, mock_cache, mock_vect):
        t = Trainer(
            model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=self.x_data, y_data=self.y_data)
        mock_vect.assert_called_once_with('harmful', None)

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.trainer.get_clf')
    @mock.patch('cherry.trainer.Trainer._write_cache')
    @mock.patch('cherry.trainer.Trainer.train')
    def test_init_no_clf(self, mock_train, mock_cache, mock_clf, mock_stop_words):
        mock_stop_words.return_value = ['first', 'second']
        t = Trainer(
            model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=self.x_data, y_data=self.y_data)
        mock_clf.assert_called_once_with('harmful', None)

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.trainer.Trainer._write_cache')
    @mock.patch('cherry.trainer.Trainer.train')
    def test_init_train(self, mock_train, mock_cache, mock_stop_words):
        mock_stop_words.return_value = ['first', 'second']
        t = Trainer(
            model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=self.x_data, y_data=self.y_data)
        mock_train.assert_called_once()

    @mock.patch('cherry.base.get_stop_words')
    @mock.patch('cherry.trainer.Trainer._write_cache')
    @mock.patch('sklearn.pipeline.Pipeline.fit')
    def test_train_fit(self, mock_fit, mock_cache, mock_stop_words):
        mock_stop_words.return_value = ['first', 'second']
        t = Trainer(
            model='harmful', vectorizer=None, vectorizer_method=None,
            clf=None, clf_method=None, x_data=self.x_data, y_data=self.y_data)
        mock_fit.assert_called_once_with(self.x_data, self.y_data)
