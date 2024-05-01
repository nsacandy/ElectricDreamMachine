"""
LSTM Classifier for Predicting Transportation.

This module implements an LSTM (Long Short-Term Memory) neural network using TensorFlow's Keras API
to predict whether individuals are transported based on various one-hot encoded features from the 'dummies_train.csv'
dataset. It includes hyperparameter optimization through manual tuning and cross-validation
to identify the best model parameters, aiming to optimize the classifier's performance on transportation prediction tasks.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import StandardScaler

__author__ = 'Thomas Lamont'
__version__ = 'Spring 2024'
__pylint__ = '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

class LSTMTransportPredictor:
    """
    A class to represent an LSTM model optimized for predicting transportation status with hyperparameter tuning.

    Attributes
    ----------
    param_grid : dict
        Grid of parameters to search for best performance.
    best_score : float
        Best score achieved during hyperparameter tuning.
    best_params : dict
        Best parameters found during hyperparameter tuning.

    Methods
    -------
    fit(self, X, y):
        Performs hyperparameter tuning and trains the model with the best parameters.

    evaluate(self, X_test, y_test):
        Evaluates the model using the testing data.
    """
    def __init__(self):
        """
        Initializes the LSTMTransportPredictor with a hyperparameter grid.
        """
        self.param_grid = {
            'neurons': [50, 100],
            'dropout_rate': [0.2, 0.3],
            'optimizer': ['adam', 'rmsprop'],
            'batch_size': [32, 64],
            'epochs': [50, 100]
        }
        self.best_score = 0
        self.best_params = {}

    def fit(self, X_train, y_train):
        """
        Performs hyperparameter tuning over the parameter grid, training models and keeping track of the best performance.
        """
        for neurons in self.param_grid['neurons']:
            for dropout_rate in self.param_grid['dropout_rate']:
                for optimizer_choice in self.param_grid['optimizer']:
                    for batch_size in self.param_grid['batch_size']:
                        for epochs in self.param_grid['epochs']:
                            print(f"Training model with: neurons={neurons}, dropout={dropout_rate}, optimizer={optimizer_choice}, batch_size={batch_size}, epochs={epochs}")
                            model = self._build_model(neurons, dropout_rate, optimizer_choice)
                            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
                            score = model.evaluate(X_train, y_train, verbose=0)[1]
                            if score > self.best_score:
                                self.best_score = score
                                self.best_params = {'neurons': neurons, 'dropout_rate': dropout_rate, 'optimizer': optimizer_choice, 'batch_size': batch_size, 'epochs': epochs}
                            print(f"Score: {score}")

    def _build_model(self, neurons, dropout_rate, optimizer_choice):
        """
        Builds and compiles an LSTM model with the specified parameters.
        """
        model = Sequential([
            LSTM(neurons, return_sequences=True, input_shape=(1, X_train.shape[2])),
            Dropout(dropout_rate),
            LSTM(neurons, return_sequences=False),
            Dropout(dropout_rate),
            Dense(neurons, activation='relu'),
            Dense(y_train.shape[1], activation='softmax')
        ])
        optimizer = Adam() if optimizer_choice == 'adam' else RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model using the testing data.
        """
        best_model = self._build_model(**self.best_params)
        best_model.fit(X_train, y_train, batch_size=self.best_params['batch_size'], epochs=self.best_params['epochs'], verbose=0)
        return best_model.evaluate(X_test, y_test, verbose=0)

# Usage
if __name__ == "__main__":
    data = pd.read_csv("dummies_train.csv")
    y = to_categorical(data['Transported'].astype(int))
    X = data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=3270)

    predictor = LSTMTransportPredictor()
    predictor.fit(X_train, y_train)
    score = predictor.evaluate(X_test, y_test)
    print(f"Test score: {score[1]}")
    print(f"Best parameters: {predictor.best_params}")