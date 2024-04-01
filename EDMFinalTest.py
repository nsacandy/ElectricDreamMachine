"""
Model Evaluation Script for Transportation Prediction.

This script evaluates the performance of three different machine learning models and their ensemble
on a transportation prediction task. The models include a Neural Network (MLPClassifier),
Logistic Regression, and a Random Forest Classifier. The dataset is split into development (dev)
and test sets, with the models trained on the dev set and evaluated on the test set.
An ensemble prediction is also made by averaging the predictions from all three models.

Usage:
- Ensure the 'dev.csv' and 'test.csv' datasets are in the same directory as this script.
- Run the script directly to evaluate the models and print their accuracies.

Dependencies:
- pandas: Used for loading and manipulating the datasets.
- sklearn: Provides tools for data preprocessing, model building, and evaluation.
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

__author__ = 'Thomas Lamont, Nic Sacandy, Dillon Emmons'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

class TransportModelEvaluator:
    """
    Evaluates Neural Network, Logistic Regression, and Random Forest models,
    and their ensemble for transportation prediction.
    """

    def __init__(self, dev_file, test_file):
        """
        Initializes the TransportModelEvaluator with datasets.
        """
        self.dev_data = pd.read_csv(dev_file)
        self.test_data = pd.read_csv(test_file)

        self.x_dev = self.dev_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
        self.y_dev = self.dev_data['Transported'].astype(int)
        self.x_test = self.test_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
        self.y_test = self.test_data['Transported'].astype(int)

        self.preprocessor = self._create_preprocessor()

    def _create_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
                                              'Spa', 'VRDeck', 'Num', 'GroupId']),
                ('cat', categorical_transformer, ['HomePlanet', 'CryoSleep', 'Destination',
                                                  'Deck', 'Side', 'VIP'])])

    def _create_nn_pipeline(self):
        return Pipeline([
            ('preprocessor', self.preprocessor), ('mlp', MLPClassifier(hidden_layer_sizes=(100,)
                , max_iter=500, alpha=0.0001, random_state=3270))])

    def _create_lr_pipeline(self):
        return Pipeline([
            ('preprocessor', self.preprocessor), ('logistic', LogisticRegression(C=100,
                max_iter=300, solver='liblinear', random_state=3270))])

    def _create_rf_pipeline(self):
        return Pipeline([
            ('preprocessor', self.preprocessor), ('random_forest',
                RandomForestClassifier(n_estimators=300, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', max_depth=10, random_state=3270))])

    def evaluate_model(self, model_pipeline):
        """
        Trains a model pipeline on the development set and evaluates its accuracy on the test set.
        """
        model_pipeline.fit(self.x_dev, self.y_dev)
        y_pred = model_pipeline.predict(self.x_test)
        return accuracy_score(self.y_test, y_pred)

    def evaluate_models(self):
        """
        Evaluates all models and prints their accuracies, including the ensemble model.
        """
        nn_pipeline = self._create_nn_pipeline()
        lr_pipeline = self._create_lr_pipeline()
        rf_pipeline = self._create_rf_pipeline()

        nn_accuracy = self.evaluate_model(nn_pipeline)
        lr_accuracy = self.evaluate_model(lr_pipeline)
        rf_accuracy = self.evaluate_model(rf_pipeline)

        nn_pred = nn_pipeline.predict(self.x_test)
        lr_pred = lr_pipeline.predict(self.x_test)
        rf_pred = rf_pipeline.predict(self.x_test)
        ensemble_pred = np.round((nn_pred + lr_pred + rf_pred) / 3.0).astype(int)
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_pred)

        print(f"Neural Network accuracy: {nn_accuracy}")
        print(f"Logistic Regression accuracy: {lr_accuracy}")
        print(f"Random Forest accuracy: {rf_accuracy}")
        print(f"Ensemble accuracy: {ensemble_accuracy}")

if __name__ == "__main__":
    evaluator = TransportModelEvaluator('dev.csv', 'test.csv')
    evaluator.evaluate_models()
    