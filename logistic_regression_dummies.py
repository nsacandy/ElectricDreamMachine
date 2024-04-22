"""
Logistic Regression Classifier for Predicting Transportation.

This module implements a logistic regression classifier using scikit-learn's LogisticRegression
for predicting whether individuals are transported based on various features from the 'dev.csv'
dataset. It performs hyperparameter tuning using GridSearchCV with StratifiedKFold cross-validation
to identify the best model parameters, addressing classification tasks within the
transportation dataset.

"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

__author__ = 'Thomas Lamont, Nic Sacandy, Dillon Emmons'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

class LogisticRegressionTransportPredictor:
    """
    A class to represent a Logistic Regression model for predicting transportation.

    Attributes
    ----------
    None

    Methods
    -------
    __init__(self):
        Initializes the LogisticRegressionTransportPredictor with default values.

    fit(self, x_train, y_train, categorical_features, numerical_features):
        Fits the Logistic Regression model to the provided training data.

    print_top5_hyperparameters(self):
        Prints the top 5 hyperparameter configurations based on cross-validation scores.
    """

    def __init__(self):
        """
        Initializes the LogisticRegressionTransportPredictor with pipeline and parameter grid.
        """
        self.pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),  # All features are now numerical
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
        Fits the Logistic Regression model to the provided training data.

        Parameters:
        x_train (DataFrame): Training features.
        y_train (Series): Training target variable.
        categorical_features (list): List of categorical feature names.
        numerical_features (list): List of numerical feature names.

        Returns:
        GridSearchCV object after fitting.
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
    train_data = pd.read_csv("dummies_train.csv")
    y_train_data = train_data['Transported'].astype(int)
    x_train_data = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])
    predictor = LogisticRegressionTransportPredictor()
    predictor.fit(x_train_data, y_train_data)
    predictor.print_top5_hyperparameters()
    