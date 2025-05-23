"""
Multilayer Perceptron Classifier for Predicting Transportation.

This module implements a Multilayer Perceptron (MLP) classifier using scikit-learn's MLPClassifier
for predicting whether individuals are transported based on various features from the 'dev.csv'
dataset. It explores different configurations of hyperparameters to optimize model performance
using StratifiedKFold cross-validation.

"""


import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import numpy as np

__author__ = 'Thomas Lamont, Nic Sacandy, Dillon Emmons'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

class MLPTransportPredictor:
    """
    A class to represent an MLP model for predicting transportation.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    __init__(self):
        Initializes the MLPTransportPredictor with default values.
    fit_and_evaluate(self, x, y):
        Fits the MLP model to the provided training data and evaluates it.
    print_results(self):
        Prints the results of the hyperparameter tuning.
    """
    def __init__(self):
        """
        Initializes the MLPTransportPredictor without specifying the data transformation
        details, which will be set up in the fit_and_evaluate method.
        """
        self.preprocessor = None
        self.results = []
        self.hyperparameters = []
        self.accuracy_scores = []

    def fit_and_evaluate(self, features, target, categorical_features, numerical_features):
        """
        Fits the MLP model to the provided training data and evaluates it.
        Parameters:
        - features: DataFrame containing the training features.
        - target: Series containing the target variable.
        - categorical_features: List of names of the categorical features.
        - numerical_features: List of names of the numerical features.
        """
        self._setup_preprocessor(categorical_features, numerical_features)
        self._evaluate_models(features, target)
    def _setup_preprocessor(self, categorical_features, numerical_features):
        """
        Sets up the data preprocessing pipeline.
        Parameters:
        - categorical_features: List of names of the categorical features.
        - numerical_features: List of names of the numerical features.
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    def _evaluate_models(self, features, target):
        """
        Evaluates the MLP model using different sets of hyperparameters.
        Parameters:
        - features: DataFrame containing the training features.
        - target: Series containing the target variable.
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)

        for hyperparams in self.hyperparameters:
            mlp_pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('mlp', MLPClassifier(**hyperparams, random_state=42))
            ])

            for train_index, test_index in skf.split(features, target):
                x_train, x_test = features.iloc[train_index], features.iloc[test_index]
                y_train, y_test = target.iloc[train_index], target.iloc[test_index]

                mlp_pipeline.fit(x_train, y_train)
                y_pred = mlp_pipeline.predict(x_test)

                accuracy = accuracy_score(y_test, y_pred)
                self.accuracy_scores.append(accuracy)

            average_accuracy = np.mean(self.accuracy_scores)
            self.results.append((hyperparams, average_accuracy))

    def print_results(self):
        """
        Prints the hyperparameter configurations and their corresponding average accuracy.
        """
        for hyperparams, accuracy in sorted(self.results, key=lambda x: x[1], reverse=True):
            print(f"Hyperparameters: {hyperparams}, Average Accuracy: {accuracy}")
    def set_hyperparameters(self, hyperparameters):
        """
        Sets the list of hyperparameters to evaluate.
        Parameters:
        - hyperparameters: List of dictionaries containing hyperparameters to try.
        """
        self.hyperparameters = hyperparameters
if __name__ == "__main__":
    train_data = pd.read_csv("dev.csv")
    x_data = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
    y_data = train_data['Transported'].astype(int)

    cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']
    num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
        'Spa', 'VRDeck', 'Num', 'GroupId']
    hyperparameters_list = [
        {'hidden_layer_sizes': (100,), 'max_iter': 300, 'alpha': 0.0001},
        {'hidden_layer_sizes': (50, 50), 'max_iter': 500, 'alpha': 0.001},
        {'hidden_layer_sizes': (100, 100), 'max_iter': 300, 'alpha': 0.00001},
        {'hidden_layer_sizes': (100, 50, 25), 'max_iter': 200, 'alpha': 0.01},
        {'hidden_layer_sizes': (50,), 'max_iter': 400, 'alpha': 0.0005},
    ]
    predictor = MLPTransportPredictor()
    predictor.set_hyperparameters(hyperparameters_list)
    predictor.fit_and_evaluate(x_data, y_data,
        cat_features, num_features)
    predictor.print_results()
    