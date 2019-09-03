import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from .base import load_data, get_vectorizer, get_clf


class Display:
    def __init__(self, model, **kwargs):
        x_data = kwargs['x_data']
        y_data = kwargs['y_data']
        if not (x_data and y_data):
            x_data, y_data = load_data(model)
        vectorizer = kwargs['vectorizer']
        vectorizer_method = kwargs['vectorizer_method']
        clf_method = kwargs['clf_method']
        clf = kwargs['clf']
        if not vectorizer:
            vectorizer = get_vectorizer(model, vectorizer_method)
        if not clf:
            clf = get_clf(model, clf_method)
        text_clf = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', clf)])
        self.display_learning_curve(text_clf, x_data, y_data)

    def display_learning_curve(self, estimator, x_data, y_data):
        title = "Learning Curves"
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        self.plot_learning_curve(estimator, title, x_data, y_data, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        import matplotlib.pyplot as plt
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
