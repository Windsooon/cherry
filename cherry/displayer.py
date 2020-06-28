import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from .base import load_all, load_data, get_vectorizer, get_clf


class Display:
    def __init__(self, model, language=None, preprocessing=None, categories=None, encoding=None, vectorizer=None,
            vectorizer_method=None, clf=None, clf_method=None, x_data=None, y_data=None):
        x_data, y_data, vectorizer, clf = load_all(
            model, language=language, preprocessing=preprocessing,
            categories=categories, encoding=encoding, vectorizer=vectorizer,
            vectorizer_method=vectorizer_method, clf=clf,
            clf_method=clf_method, x_data=x_data, y_data=y_data)
        self.display_learning_curve(vectorizer, clf, x_data, y_data)

    def display_learning_curve(self, vectorizer, clf, x_data, y_data):
        title = "Learning Curves"
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        self.plot_learning_curve(text_clf, title, x_data, y_data, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        # From https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        print('Drawing curve, depending on your datasets size, this may take several minutes to several hours.')
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        plt.show()
