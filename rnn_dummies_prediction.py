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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


# Load the dataset
data = pd.read_csv("dummies_train.csv")
y_train = to_categorical(data['Transported'].astype(int))  # One-hot encode the target
X_train = data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])


test_data = pd.read_csv("dummies_test.csv")
X_test = test_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and testing sets
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

def create_model(neurons=50, dropout_rate=0.2):
    model = Sequential()
    model.add(SimpleRNN(neurons, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(neurons, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create the KerasClassifier
model = create_model(neurons=50, dropout_rate=0.2)
model.fit(X_train_scaled, y_train, batch_size=32, epochs=100, verbose=1)

# Make predictions on the test data
predictions_proba = model.predict(X_test_scaled)
predictions = np.argmax(predictions_proba, axis=1)  # Get the index of the max probability

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions
})

# Convert numeric predictions to 'TRUE' or 'FALSE'
predictions_df['Transported'] = predictions_df['Transported'].map({0: 'False', 1: 'True'})

# Save the predictions to a CSV file
predictions_df.to_csv("rnn_predictions.csv", index=False, header=True)

print("Predictions have been saved to 'rnn_predictions.csv'")