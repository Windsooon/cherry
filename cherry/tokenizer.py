import os
from .config import DATA_DIR
from .exceptions import LanguageNotFoundError


def get_tokenizer(**kwargs):
    token = Token(**kwargs)
    return token.tokenizer


class Token:
    def __init__(self, **kwargs):
        self.stop_word = self._get_stop_word(kwargs['lan'])
        if kwargs['split']:
            self.tokenizer = kwargs['split'](kwargs['text'], self.stop_word)
        else:
            self.tokenizer = self._get_tokenizer(kwargs['text'], kwargs['lan'])

    def _get_tokenizer(self, text, lan):
        if lan == 'Chinese':
            import jieba
            return [
                t for t in jieba.cut(text) if len(t) > 1
                and t not in self.stop_word]
        elif lan == 'English':
            return [
                t for t in text.lower().split(' ')
                if t not in self.stop_word]

    def _get_stop_word(self, lan):
        try:
            lan_data_path = os.path.join(
                DATA_DIR, ('data/' + lan + '/'))
            stop_word_path = os.path.join(
                lan_data_path, 'stop_word.dat')
            with open(stop_word_path, encoding='utf-8') as f:
                stop_word = [l[:-1] for l in f.readlines()]
        except IOError:
            error = (
                'Language {0} not found'.format(lan))
            raise LanguageNotFoundError(error)
        return stop_word
