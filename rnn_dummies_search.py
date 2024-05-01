"""
Logistic Regression Classifier for Predicting Transportation.

This module implements a logistic regression classifier using scikit-learn's LogisticRegression
for predicting whether individuals are transported based on various features from the 'dev.csv'
dataset. It performs hyperparameter tuning using GridSearchCV with StratifiedKFold cross-validation
to identify the best model parameters, addressing classification tasks within the
transportation dataset.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("dummies_train.csv")
y = to_categorical(data['Transported'].astype(int))  # One-hot encode the target
X = data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=3270)


# Define parameter grid
param_grid = {
    'neurons': [50, 100],
    'dropout_rate': [0.2, 0.3],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64],
    'epochs': [50, 100],
}

def create_model(neurons=50, dropout_rate=0.2, optimizer_choice='adam'):
    model = Sequential()
    model.add(SimpleRNN(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(neurons, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))  # Assuming y was one-hot encoded
    if optimizer_choice == 'adam':
        optimizer = Adam()
    else:
        optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create the KerasClassifier
best_score = 0
best_params = None

for neurons in param_grid['neurons']:
    for dropout_rate in param_grid['dropout_rate']:
        for optimizer_choice in param_grid['optimizer']:
            for batch_size in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    print(f"Training model with: neurons={neurons}, dropout={dropout_rate}, optimizer={optimizer_choice}, batch_size={batch_size}, epochs={epochs}")
                    model = create_model(neurons, dropout_rate, optimizer_choice)
                    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
                    score = model.evaluate(X_test, y_test, verbose=0)[1]  
                    print(f"Score: {score}")
                    if score > best_score:
                        best_score = score
                        best_params = {'neurons': neurons, 'dropout_rate': dropout_rate, 'optimizer': optimizer_choice, 'batch_size': batch_size, 'epochs': epochs}

print(f"Best score: {best_score}")
print(f"Best params: {best_params}")