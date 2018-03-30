import unittest
from unittest import mock
import jieba
from cherry.trainer import Trainer


class TrainerTest(unittest.TestCase):

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_test_data_num_with_custom_split(self, mock_cut):
        mock_cut.return_value = ['警方', '发布', '了', '最新消息']

        def split_function(text):
            stop_word = ['但是']
            return [
                t for t in jieba.cut(text) if len(t) > 1
                and t not in stop_word]
        trainer = Trainer(test_num=0, lan='Chinese', split=split_function)
        self.assertEqual(
            trainer.test_num, 0)

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_test_data_num(self, mock_cut):
        trainer = Trainer(test_num=1, lan='Chinese', split=None)
        self.assertEqual(
            trainer.test_num, 1)

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_empty_vocab_list(self, mock_cut):
        trainer = Trainer(test_num=1, lan='Chinese', split=None)
        self.assertEqual(
            trainer.vocab_set, set())
