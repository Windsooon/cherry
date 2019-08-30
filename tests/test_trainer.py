import os
import unittest
from unittest import mock
from sklearn.pipeline import Pipeline
from cherry.config import DEFAULT_VECTORIZER, DEFAULT_CLF
from cherry.base import DATA_DIR
from cherry.trainer import Trainer


class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.x_data = [1, 2]
        self.y_data = [0, 1]

    @mock.patch('cherry.trainer.Trainer.train')
    def test_write_cache(self, mock_train):
        '''
        TODO: use create special cache files for testing
        '''
        mock_train.return_value = DEFAULT_VECTORIZER, DEFAULT_CLF
        t = Trainer(vectorizer=DEFAULT_VECTORIZER, clf=DEFAULT_CLF, x_data=self.x_data, y_data=self.y_data)
        self.assertTrue(os.path.exists(os.path.join(DATA_DIR, 'trained.pkl')))
        self.assertTrue(os.path.exists(os.path.join(DATA_DIR, 've.pkl')))

    @mock.patch('sklearn.pipeline.Pipeline.fit')
    def test_Trainer_init(self, mock_fit):
        mock_fit.return_value = DEFAULT_VECTORIZER, DEFAULT_CLF
        x_data = [1,2]
        y_data = [0,1]
        t = Trainer(vectorizer=DEFAULT_VECTORIZER, clf=DEFAULT_CLF, x_data=x_data, y_data=y_data)
        vectorizer, clf = t.train(DEFAULT_VECTORIZER, DEFAULT_CLF, x_data, y_data)
        self.assertEqual(vectorizer, DEFAULT_VECTORIZER)
        self.assertEqual(clf, DEFAULT_CLF)
