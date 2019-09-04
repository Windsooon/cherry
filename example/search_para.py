from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

import cherry
import numpy as np

parameters = {
    'clf__alpha': [0.1, 0.5, 1],
    'clf__fit_prior': [True, False]
}


cherry.search('harmful', parameters, cv=10, n_jobs=-1)
