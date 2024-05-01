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

__author__ = 'Thomas Lamont, Nic Sacandy, Dillon Emmons'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'

train_data = pd.read_csv("preprocessed_train.csv")
test_data = pd.read_csv("preprocessed_test.csv")

X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
y_train = train_data['Transported'].astype(int)

X_test = test_data.drop(columns=['PassengerId', 'Name', 'Cabin'])

categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
    'Spa', 'VRDeck', 'Num', 'GroupId']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

nn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,),
        max_iter=500, alpha=0.0001, random_state=3270))])

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logistic', LogisticRegression(C=100, max_iter=300, solver='liblinear', random_state=3270))])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('random_forest', RandomForestClassifier(n_estimators=300, min_samples_split=2,
        min_samples_leaf=1, max_features='sqrt', max_depth=10, random_state=3270))])

nn_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

unknown_preprocessed = preprocessor.transform(test_data.drop(columns=['PassengerId',
    'Name', 'Cabin']))
nn_pred = nn_pipeline.named_steps['mlp'].predict(unknown_preprocessed)
lr_pred = lr_pipeline.named_steps['logistic'].predict(unknown_preprocessed)
rf_pred = rf_pipeline.named_steps['random_forest'].predict(unknown_preprocessed)

ensemble_pred = (nn_pred.astype(int) + lr_pred.astype(int) + rf_pred.astype(int)) >= 2

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': ensemble_pred})

output['Transported'] = output['Transported'].map({True: 'True', False: 'False'})

output.to_csv("RF2Prediction.csv", index=False)

print("Ensemble predictions exported to EnsemblePredictions.csv")
