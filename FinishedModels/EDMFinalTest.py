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

# Define a function to evaluate a given model and hyperparameters on dev and test sets
def evaluate_model(model_pipeline, X_dev, y_dev, X_test, y_test):
    # Train the model on the entire development set
    model_pipeline.fit(X_dev, y_dev)

    # Predict on the test set
    y_pred = model_pipeline.predict(X_test)

    # Calculate accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Load the development and test data
dev_data = pd.read_csv("dev.csv")
test_data = pd.read_csv("test.csv")

# Separate features and target variable for development and test data
X_dev = dev_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
y_dev = dev_data['Transported'].astype(int)

X_test = test_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
y_test = test_data['Transported'].astype(int)

# Define categorical and numerical features
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'GroupId']

# Create preprocessing pipelines for both numeric and categorical data
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

# Neural Network hyperparameters and pipeline
nn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=0.0001, random_state=3270))])

# Logistic Regression hyperparameters and pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logistic', LogisticRegression(C=100, max_iter=300, solver='liblinear', random_state=3270))])

# Random Forest hyperparameters and pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('random_forest', RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=10, random_state=3270))])

# Evaluate models
nn_accuracy = evaluate_model(nn_pipeline, X_dev, y_dev, X_test, y_test)
lr_accuracy = evaluate_model(lr_pipeline, X_dev, y_dev, X_test, y_test)
rf_accuracy = evaluate_model(rf_pipeline, X_dev, y_dev, X_test, y_test)


nn_pred = nn_pipeline.predict(X_test)
lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_pipeline.predict(X_test)

# Ensemble prediction: True if at least two models predict True
ensemble_pred = np.round((nn_pred.astype(int) + lr_pred.astype(int) + rf_pred.astype(int)) / 3)


ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
# Print accuracies
print(f"Neural Network accuracy: {nn_accuracy}")
print(f"Logistic Regression accuracy: {lr_accuracy}")
print(f"Random Forest accuracy: {rf_accuracy}")
print(f"Ensemble accuracy: {ensemble_accuracy}")