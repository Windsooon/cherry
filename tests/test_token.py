import unittest
from unittest import mock
from cherry.models import Token
from cherry.exceptions import LanguageNotFoundError


class TokenTest(unittest.TestCase):

    @mock.patch('jieba.cut')
    @mock.patch('builtins.open', mock.mock_open(read_data='abcdefghijk\n'))
    def test_chinese_stop_word(self, mock_cut):
        mock_cut.return_value = ['警方', '召开']
        text = '警方召开。'
        token = Token(text=text, lan='Chinese', split=None)
        self.assertEqual(token.stop_word, ['abcdefghijk'])
