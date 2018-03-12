import unittest
from unittest import mock
from cherry.models import Token


class TokenTest(unittest.TestCase):

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='但是\n'))
    def test_chinese_stop_word(self, mock_cut):
        mock_cut.return_value = ['警方', '发布', '最新', '消息']
        text = '警方发布了最新消息。'
        token = Token(text=text, lan='Chinese', split=None)
        self.assertEqual(token.stop_word, ['但是'])

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='但是\n'))
    def test_get_tokenizer(self, mock_cut):
        mock_cut.return_value = ['警方', '发布', '最新', '消息']
        text = '警方发布了最新消息。'
        token = Token(text=text, lan='Chinese', split=None)
        self.assertEqual(token.tokenizer, ['警方', '发布', '最新', '消息'])
