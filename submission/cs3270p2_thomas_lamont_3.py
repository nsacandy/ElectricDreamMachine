"""
Random Forest Classifier for Predicting Transportation.

This module implements a Random Forest classifier using scikit-learn's RandomForestClassifier
for predicting whether individuals are transported based on various one-hot
encoded features from the 'dummies_train.csv'
dataset. It employs hyperparameter optimization using RandomizedSearchCV
with StratifiedKFold cross-validation to enhance model performance.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

__author__ = 'Thomas Lamont'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

class RandomForestTransportPredictor:
    """
    A class to represent a Random Forest model for predicting transportation.

    Attributes
    ----------
    pipeline : Pipeline
        Defines the steps for preprocessing and classification.
    param_grid_rf : dict
        Specifies the grid of hyperparameters for RandomizedSearchCV.

    Methods
    -------
    fit_and_optimize(x_train, y_train):
        Fits the model to the training data and optimizes hyperparameters.
    print_top5_results():
        Prints the top 5 hyperparameter configurations based on cross-validation scores.
    """

    def __init__(self):
        """
        Initializes the RandomForestTransportPredictor with a pipeline for standard scaling and
        a Random Forest classifier, along with a parameter grid for optimizing the classifier.
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, []),
            ('cat', categorical_transformer, [])])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
            ('classifier', RandomForestClassifier(random_state=3270))])

        self.param_grid_rf = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
        self.cv_results_ = None
        self.random_search_rf = None

    def fit_and_optimize(self, x_train, y_train):
        """
        Fits the model using the provided training data and optimizes hyperparameters.
        """
        self.preprocessor = StandardScaler()

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=3270))
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)
        self.random_search_rf = RandomizedSearchCV(
        self.pipeline, self.param_grid_rf, n_iter=100, cv=skf, verbose=2,
            n_jobs=-1, random_state=3270)
        self.random_search_rf.fit(x_train, y_train)

        self.cv_results_ = self.random_search_rf.cv_results_

    def print_top5_results(self):
        """
        Prints the top 5 hyperparameter configurations based on cross-validation scores.
        """
        top5_indices = np.argsort(-self.cv_results_['mean_test_score'])[:5]

        print("Top 5 parameter combinations:")
        for rank, index in enumerate(top5_indices, start=1):
            print(f"Rank: {rank}")
            print(f"Score: {self.cv_results_['mean_test_score'][index]}")
            print(f"Parameters: {self.cv_results_['params'][index]}\n")

if __name__ == "__main__":
    train_data = pd.read_csv("cs3270p2_thomas_lamont_train1.csv")
    y_train_data = train_data['Transported'].astype(int)
    x_train_data = train_data.drop(columns=['Transported', 'PassengerId', 'Name',
        'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E',
        'Deck_F', 'Deck_G', 'Deck_T', 'Num'])
    predictor = RandomForestTransportPredictor()
    predictor.fit_and_optimize(x_train_data, y_train_data)
    predictor.print_top5_results()
    