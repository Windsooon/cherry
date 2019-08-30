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
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from .base import stop_words

def tokenizer(text):
    '''
    You can use your own tokenizer function here, by default,
    this function only work for chinese
    '''
    # For English, try:
    # from nltk.tokenize import word_tokenize
    # return [t.lower() for t in word_tokenize(text) if len(t) > 1]
    return [t for t in jieba.cut(text) if len(t) > 1]


FILENAME = 'chinese_classify.txt'
STOPWORDS = 'stop_words_' + FILENAME
DEFAULT_VECTORIZER = CountVectorizer(tokenizer=tokenizer, stop_words=stop_words(STOPWORDS))
DEFAULT_CLF = MultinomialNB(alpha=0.1)
