from .tokenizer import get_tokenizer


class Result:
    def __init__(self, **kwargs):
        self.token = get_tokenizer(**kwargs)

    def _calculate_ps(self, text):
        pass

    @property
    def get_token(self):
        return self.token
