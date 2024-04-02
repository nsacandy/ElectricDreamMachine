"""
Multinomial Naive Bayes Classifier for Predicting Transportation.

This module implements a Multinomial Naive Bayes classifier using scikit-learn's MultinomialNB
for predicting whether individuals are transported based on various features from the 'dev.csv'
dataset. It performs hyperparameter tuning using GridSearchCV with StratifiedKFold cross-validation
to identify the best model parameters.

"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np

__author__ = 'Thomas Lamont, Nic Sacandy, Dillon Emmons'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

class MultinomialNBTransportPredictor:
    """
    A class to represent a Multinomial Naive Bayes model for predicting transportation.

    Attributes:
    ----------
    None

    Methods:
    -------
    __init__(self):
        Initializes the MultinomialNBTransportPredictor with default values.

    fit(self, x_train, y_train, categorical_features, numerical_features):
        Fits the Multinomial Naive Bayes model to the provided training data.

    print_top5_hyperparameters(self):
        Prints the top 5 hyperparameter configurations based on cross-validation scores.
    """

    def __init__(self):
        """
        Initializes the MultinomialNBTransportPredictor with pipeline and parameter grid.
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, []),
                ('cat', categorical_transformer, [])])

        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('classifier', MultinomialNB())])

        self.param_grid = {
            'classifier__alpha': np.logspace(0, -9, num=100),
            'classifier__fit_prior': [True, False]
        }
        self.cv_results_ = None
        self.best_estimator_ = None

    def fit(self, x_train, y_train, categorical_features, numerical_features):
        """
        Fits the Multinomial Naive Bayes model to the provided training data.

        Parameters:
        x_train (DataFrame): Training features.
        y_train (Series): Training target variable.
        categorical_features (list): List of categorical feature names.
        numerical_features (list): List of numerical feature names.

        Returns:
        GridSearchCV object after fitting.
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor.transformers = [
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)]

        self.pipeline.steps[0] = ('preprocessor', self.preprocessor)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)
        grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=skf,
            scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(x_train, y_train)

        self.cv_results_ = grid_search.cv_results_
        self.best_estimator_ = grid_search.best_estimator_

        return grid_search

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
    train_data = pd.read_csv("dev.csv")
    y_train_data = train_data['Transported'].astype(int)
    x_train_data = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
    binary_features = ['CryoSleep', 'VIP']
    for column in binary_features:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])

    cat_features = ['HomePlanet', 'Deck', 'Side', 'Destination']
    num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']

    predictor = MultinomialNBTransportPredictor()
    predictor.fit(x_train_data, y_train_data, cat_features, num_features)
    predictor.print_top5_hyperparameters()
    