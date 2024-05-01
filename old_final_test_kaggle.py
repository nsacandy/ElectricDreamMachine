"""
Transportation Prediction Model Evaluation and Prediction Export.

This script is designed to train three different machine learning models: Neural Network
(MLPClassifier), Logistic Regression, and Random Forest Classifier, on a preprocessed training
dataset. It then makes predictions on a preprocessed test dataset, creates an ensemble prediction
from the three models, and exports these predictions to a CSV file. The script assumes the existence
of 'preprocessed_train.csv' and 'preprocessed_test.csv' files, which are the preprocessed versions
of the training and test datasets from the Kaggle competition.

The script employs sklearn pipelines for data preprocessing, including imputation, scaling, and
one-hot encoding, followed by model training. The ensemble prediction is made by taking the
majority vote from the three models' predictions.

Usage:
- Ensure 'preprocessed_train.csv' and 'preprocessed_test.csv' are available in the working directory
- Run the script to train the models, make ensemble predictions, and export the predictions
  to 'EnsemblePredictions.csv'.

Dependencies:
- pandas: For data manipulation and CSV I/O.
- sklearn: For data preprocessing, model training, and prediction.

Output:
- 'RF2Prediction.csv': Contains the ensemble predictions for the test dataset, with two columns:
    PassengerId and Transported.
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

__author__ = 'Thomas Lamont, Nic Sacandy, Dillon Emmons'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

train_data = pd.read_csv("dummies_train.csv")
test_data = pd.read_csv("dummies_test.csv")

X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name','Cabin'])
y_train = train_data['Transported'].astype(int)

X_test = test_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001, random_state=3270)

logreg = LogisticRegression(C=1, max_iter=300, solver='lbfgs', random_state=3270)

rf = RandomForestClassifier(n_estimators=200, min_samples_split=10, min_samples_leaf=4, 
                            max_features='sqrt', max_depth=10, random_state=3270)
                            
mlp_pipeline = Pipeline(steps=[('scaler', scaler), ('mlp', mlp)])
logreg_pipeline = Pipeline(steps=[('scaler', scaler), ('logistic', logreg)])
rf_pipeline = Pipeline(steps=[('scaler', scaler), ('random_forest', rf)])

mlp_pipeline.fit(X_train, y_train)
logreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# Make predictions using each model
mlp_pred = mlp_pipeline.predict(X_test)
logreg_pred = logreg_pipeline.predict(X_test)
rf_pred = rf_pipeline.predict(X_test)

ensemble_pred = np.round((mlp_pred + logreg_pred + rf_pred) / 3.0).astype(int)

# Convert predictions to 'True' or 'False'
ensemble_pred = np.where(ensemble_pred == 1, 'True', 'False')

# Prepare the output DataFrame
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': ensemble_pred})

# Save the predictions to a CSV file
output.to_csv("updated_ensemble_predictions_no_drop.csv", index=False)

print("Ensemble predictions exported to updated_ensemble_predictions.csv")