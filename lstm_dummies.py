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
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("dummies_train.csv")
y = to_categorical(data['Transported'].astype(int))  # One-hot encode the target
X = data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input to be 3D [samples, timesteps, features] required by LSTM layer
# Assuming you can shape your data into sequences; otherwise, you'll need a different approach
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy*100:.2f}')