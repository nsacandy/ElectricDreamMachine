"""
This script implements an ensemble prediction model combining traditional machine learning techniques with deep learning models
to predict whether passengers are transported based on various features from a CSV dataset.
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

__author__ = 'Thomas Lamont'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

train_data = pd.read_csv("cs3270p2_thomas_lamont_train1.csv")
test_data = pd.read_csv("cs3270p2_thomas_lamont_test1.csv")

X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])
y_train = train_data['Transported'].astype(int)

X_test = test_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Num'])

y_train_keras = to_categorical(y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_scaled_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001, random_state=3270)

logreg = LogisticRegression(C=1, max_iter=300, solver='lbfgs', random_state=3270)

rf = RandomForestClassifier(n_estimators=200, min_samples_split=10, min_samples_leaf=4, 
                            max_features='sqrt', max_depth=10, random_state=3270)
                            
def create_rnn_model():
    """Creates and compiles an RNN model."""
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(X_train_scaled_reshaped.shape[1], X_train_scaled_reshaped.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))  # 2 for binary classification
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

def create_lstm_model():
    """Creates and compiles an LSTM model."""
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train_scaled_reshaped.shape[1], X_train_scaled_reshaped.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 for binary classification
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model                           
                            
mlp_pipeline = Pipeline(steps=[('scaler', scaler), ('mlp', mlp)])
logreg_pipeline = Pipeline(steps=[('scaler', scaler), ('logistic', logreg)])
rf_pipeline = Pipeline(steps=[('scaler', scaler), ('random_forest', rf)])

mlp_pipeline.fit(X_train, y_train)
logreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

rnn_model = create_rnn_model()
lstm_model = create_lstm_model()

rnn_model.fit(X_train_scaled_reshaped, y_train_keras, batch_size=32, epochs=100, verbose=1)
lstm_model.fit(X_train_scaled_reshaped, y_train_keras, batch_size=32, epochs=50, verbose=1)

mlp_pred = mlp_pipeline.predict(X_test)
logreg_pred = logreg_pipeline.predict(X_test)
rf_pred = rf_pipeline.predict(X_test)

rnn_predictions_proba = rnn_model.predict(X_test_scaled_reshaped)
lstm_predictions_proba = lstm_model.predict(X_test_scaled_reshaped)
rnn_pred = np.argmax(rnn_predictions_proba, axis=-1)
lstm_pred = np.argmax(lstm_predictions_proba, axis=-1)

ensemble_preds = np.array([mlp_pred, logreg_pred, rf_pred, rnn_pred, lstm_pred])
ensemble_majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=ensemble_preds)

ensemble_majority_vote = np.where(ensemble_majority_vote == 1, 'True', 'False')

submission_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': ensemble_majority_vote})

submission_df.to_csv("final_ensemble_submission.csv", index=False)

print("Ensemble submission saved to ensemble_submission.csv")