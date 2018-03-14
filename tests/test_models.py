import unittest
from unittest import mock
from cherry.models import Trainer
from cherry.config import POSITIVE


class ModelsTest(unittest.TestCase):

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_read_files(self, mock_cut):
        trainer = Trainer(
            test_num=0, test_mode=False,
            lan='Chinese', split=None)
        self.assertEqual(
            trainer.data_list,
            [(0, '警方发布了最新消息\n'), (1, '警方发布了最新消息\n')])
        self.assertEqual(
            trainer.CLASSIFY,
            ['gamble.dat', 'normal.dat'])
        self.assertEqual(
            trainer.data_len, 2)

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_data_num_test_mode_false(self, mock_cut):
        trainer = Trainer(
            test_num=0, test_mode=False, lan='Chinese',
            split=None, positive=POSITIVE, binary=True)
        self.assertEqual(
            trainer.test_num, 0)

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_data_num_test_mode_true(self, mock_cut):
        trainer = Trainer(
            test_num=0, test_mode=True, lan='Chinese',
            split=None, positive=POSITIVE, binary=True)
        self.assertEqual(
            trainer.test_num, 0)

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_data_num_test_mode_true_num_10(self, mock_cut):
        trainer = Trainer(
            test_num=1, test_mode=True, lan='Chinese',
            split=None, positive=POSITIVE, binary=True)
        self.assertEqual(
            trainer.test_num, 1)

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_train_data_value(self, mock_cut):
        trainer = Trainer(
            test_num=1, test_mode=True, lan='Chinese',
            split=None, positive=POSITIVE, binary=True)
        self.assertEqual(
            trainer.train_data + trainer.test_data,
            [(0, '警方发布了最新消息\n'), (1, '警方发布了最新消息\n')])

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='警方发布了最新消息\n'))
    def test_vocab_list(self, mock_cut):
        trainer = Trainer(
            test_num=1, test_mode=True, lan='Chinese',
            split=None, positive=POSITIVE, binary=True)
        self.assertEqual(
            trainer.vocab_list, [])

    def test_ps_vector(self, mock_cut):
        trainer = Trainer(
            test_num=0, test_mode=False, lan='Chinese',
            split=None, positive=POSITIVE, binary=True)
        self.assertEqual(
            trainer.ps_vector, [])
