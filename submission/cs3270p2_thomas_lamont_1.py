"""
Logistic Regression Classifier for Predicting Transportation.

This module implements a logistic regression classifier using scikit-learn's LogisticRegression
for predicting whether individuals are transported based on various one-hot
encoded features from the 'dummies_train.csv'
dataset. It includes hyperparameter tuning using GridSearchCV with StratifiedKFold cross-validation
to identify the best model parameters, aiming to optimize
the classifier's performance on transportation prediction tasks.

"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

__author__ = 'Thomas Lamont'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

class LogisticRegressionTransportPredictor:
    """
    A class to represent a Logistic Regression model optimized for predicting transportation status.

    Attributes
    ----------
    pipeline : Pipeline
        A processing pipeline that standardizes the data and applies logistic regression.

    Methods
    -------
    __init__(self):
        Initializes the LogisticRegressionTransportPredictor with
        a data processing pipeline and a parameter grid for model optimization.

    fit(self, x_train, y_train):
        Fits the Logistic Regression model to the provided
        training data and performs hyperparameter tuning.

    print_top5_hyperparameters(self):
        Prints the top 5 hyperparameter configurations based on cross-validation scores.
    """

    def __init__(self):
        """
        Initializes the LogisticRegressionTransportPredictor with a
        pipeline for standardizing the data and optimizing logistic regression parameters.
        """
        self.pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=3270))
        ])

        self.param_grid_lr = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'classifier__max_iter': [100, 200, 300]
        }
        self.cv_results_ = None
        self.best_estimator_ = None

    def fit(self, x_train, y_train):
        """
        Displays the five best sets of parameters found during
        the hyperparameter tuning process, according to their cross-validation scores.
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)
        grid_search_lr = GridSearchCV(self.pipeline, self.param_grid_lr,
                                      cv=skf, verbose=1, n_jobs=-1, scoring='accuracy')
        grid_search_lr.fit(x_train, y_train)

        self.cv_results_ = grid_search_lr.cv_results_
        self.best_estimator_ = grid_search_lr.best_estimator_

        return grid_search_lr

    def print_top5_hyperparameters(self):
        """
        Prints the top 5 hyperparameter configurations based on cross-validation scores.
        """
        top5_indices = np.argsort(-self.cv_results_['mean_test_score'])[:5]
        print("Top 5 parameter combinations:")
        for rank, index in enumerate(top5_indices, start=1):
            print(f"Rank: {rank}")
            print(f"Score: {self.cv_results_['mean_test_score'][index]}")
            print(f"Parameters: {self.cv_results_['params'][index]}\n")


# Usage
if __name__ == "__main__":
    train_data = pd.read_csv("cs3270p2_thomas_lamont_train1.csv")
    y_train_data = train_data['Transported'].astype(int)
    x_train_data = train_data.drop(columns=
        ['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B'
        , 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])
    predictor = LogisticRegressionTransportPredictor()
    predictor.fit(x_train_data, y_train_data)
    predictor.print_top5_hyperparameters()
    