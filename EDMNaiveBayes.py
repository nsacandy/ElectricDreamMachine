import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Load the data
train_data = pd.read_csv("dev.csv")

# Define a function to extract cabin features
def process_cabin(data):
    # Split the Cabin column
    cabins = data['Cabin'].str.split('/', expand=True)
    # Name the columns
    cabins.columns = ['Deck', 'Num', 'Side']
    # Convert 'Num' to numeric
    cabins['Num'] = pd.to_numeric(cabins['Num'], errors='coerce')
    return cabins

# Process the cabin data
cabin_data = process_cabin(train_data)

# Combine the cabin data back into the original dataframe
train_data = pd.concat([train_data, cabin_data], axis=1)

# Now drop the original 'Cabin' column
train_data = train_data.drop(columns=['Cabin'])

# Encode binary categorical features as 0 or 1
binary_features = ['CryoSleep', 'VIP']
for column in binary_features:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])

# Separate the target variable and drop non-predictive columns
y_train = train_data['Transported'].astype(int)
X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name'])  # Assuming 'Name' is non-predictive

# Define categorical and numerical features
categorical_features = ['HomePlanet', 'Deck', 'Side', 'Destination']  # Updated to include Deck and Side
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']  # Added 'Num'

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Using median to handle numerical NaNs
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Create a preprocessing and modeling pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', GaussianNB())])

# Parameter grid for GaussianNB
param_grid = {
    'classifier__var_smoothing': np.logspace(0,-9, num=5)
}

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)

# Setup GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=1)

# Execute grid search
grid_search.fit(X_train, y_train)

# Get the best score and parameters
best_score = grid_search.best_score_
best_params = grid_search.best_params_

# Print the best score and parameters
print(f"Best score: {best_score}")
print(f"Best parameters: {best_params}")