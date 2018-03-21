import unittest
from unittest.mock import patch, MagicMock
import cherry


class ApiTest(unittest.TestCase):

    def test_classify(self):
        pass

    @patch('cherry.trainer.Trainer')
    def test_trainer(self, Trainer):
        pass
