# -*- coding: utf-8 -*-

"""
cherry.config
~~~~~~~~~~~~
Base config of cherry classify
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from .base import tokenizer, stop_words

PREFIX = 'chinese_classify'
DEFAULT_VECTORIZER = CountVectorizer(tokenizer=tokenizer, stop_words=stop_words(PREFIX))
DEFAULT_CLF = MultinomialNB()
NGRAM = (1, 1)
MAX_DF = 1.0
MIN_DF = 1
