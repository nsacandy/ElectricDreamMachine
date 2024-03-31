import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import numpy as np

# Load the data
train_data = pd.read_csv("dev.csv")

# Separate features and target variable for training data
X = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
y = train_data['Transported'].astype(int)

# Define categorical and numerical features
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'GroupId']

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values using mean strategy
    ('scaler', StandardScaler())])  # Scale the data

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values for categorical data
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])  # One hot encode the categorical data

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)

# List of hyperparameters to try
hyperparameters_list = [
    {'hidden_layer_sizes': (100,), 'max_iter': 300, 'alpha': 0.0001},
    {'hidden_layer_sizes': (50, 50), 'max_iter': 500, 'alpha': 0.001},
    {'hidden_layer_sizes': (100, 100), 'max_iter': 300, 'alpha': 0.00001},
    {'hidden_layer_sizes': (100, 50, 25), 'max_iter': 200, 'alpha': 0.01},
    {'hidden_layer_sizes': (50,), 'max_iter': 400, 'alpha': 0.0005},
]

# Results list to store hyperparameters and their corresponding average accuracies
results = []

# Iterate over different sets of hyperparameters
for hyperparams in hyperparameters_list:
    # Pipeline with preprocessing and MLPClassifier
    mlp_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('mlp', MLPClassifier(**hyperparams, solver='adam', verbose=10, random_state=42, tol=0.0001))
    ])

    # List to store accuracy scores for each fold
    accuracy_scores = []

    # Perform 5-fold cross-validation
    for train_index, test_index in skf.split(X, y):
        # Split the data into training and testing sets
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Train the classifier
        mlp_pipeline.fit(X_train_fold, y_train_fold)

        # Predict labels for the test set
        y_pred = mlp_pipeline.predict(X_test_fold)

        # Compute accuracy and store it
        accuracy = accuracy_score(y_test_fold, y_pred)
        accuracy_scores.append(accuracy)

    # Calculate the average accuracy across all folds
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    
    # Store the results
    results.append((hyperparams, average_accuracy))

# After all iterations are done, print all the results
print("All results:")
for hyperparams, average_accuracy in results:
    print(f"Hyperparameters: {hyperparams}")
    print(f"Average Accuracy: {average_accuracy}\n")
