# -*- coding: utf-8 -*-

"""
cherry.api
~~~~~~~~~~~~
This module implements the cherry API.
:copyright: (c) 2018-2019 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from .base import load_data
from .trainer import Trainer
from .classifyer import Classify
from .performancer import Performance
from .searcher import Search
from .displayer import Display


def classify(model, text):
    '''
    Return a Classify object which contains *probability* and *word_list*

    input: text (list of string): the text to be classified
           number of word list (int): how many word should be list in the word list
    output: Classify (Classify object)
    '''
    return Classify(model=model, text=text)

def train(model, language='English', preprocessing=None, categories=None, encoding='utf-8',
        vectorizer=None, vectorizer_method='Count', clf=None, clf_method='MNB', x_data=None, y_data=None):
    '''
    Train the `model` inside data directory

    *model (string): model name of the training dataset (e.g. 'harmful')
    language (string): The language of the data
    vectorizer (BaseEstimator object): feature extraction method, CountVectorizer, TfidfVectorizer, HashingVectorizer etc.
    clf (Classifier object): Classifier object, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier etc.
    x_data (array): training data
    y_data (array): training label
    '''
    return Trainer(
            model, language=language, preprocessing=preprocessing, encoding=encoding,
            categories=categories, vectorizer=vectorizer, vectorizer_method=vectorizer_method,
            clf=clf, clf_method=clf_method, x_data=x_data, y_data=y_data)

def performance(model, language='English', preprocessing=None, categories=None, encoding='utf-8',
        vectorizer=None, vectorizer_method='Count', clf=None, clf_method='MNB', x_data=None,
        y_data=None, n_splits=10, output='Stdout'):
    '''
    Calculate scores from the models
    '''
    return Performance(
            model, language=language, preprocessing=preprocessing, encoding=encoding,
            categories=categories, vectorizer=vectorizer, vectorizer_method=vectorizer_method,
            clf=clf, clf_method=clf_method, x_data=x_data, y_data=y_data,
            n_splits=n_splits, output=output)

def search(model, parameters, language='English', preprocessing=None, categories=None, encoding='utf-8',
        vectorizer=None, vectorizer_method='Count', clf=None, clf_method='MNB', x_data=None,
        y_data=None, method='RandomizedSearchCV', cv=3, n_jobs=-1):
    '''
    Search the best parameters
    '''
    return Search(
            model, parameters, language=language, preprocessing=preprocessing, encoding=encoding,
            categories=categories, vectorizer=vectorizer,
            vectorizer_method=vectorizer_method, clf=clf, clf_method=clf_method,
            method=method, x_data=x_data, y_data=y_data, cv=cv, n_jobs=n_jobs)

def display(model, language='English', preprocessing=None, categories=None, encoding='utf-8',
        vectorizer=None, vectorizer_method='Count', clf=None, clf_method='MNB', x_data=None, y_data=None):
    '''
    Display the learning curve
    '''
    return Display(
            model, language=language, preprocessing=preprocessing, encoding=encoding,
            categories=categories, vectorizer=vectorizer, vectorizer_method=vectorizer_method,
            clf=clf, clf_method=clf_method, x_data=x_data, y_data=y_data)
