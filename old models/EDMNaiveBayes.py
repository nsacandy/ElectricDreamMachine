import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Load the data
train_data = pd.read_csv("preprocessed_train.csv")

# Encode binary categorical features as 0 or 1
binary_features = ['CryoSleep', 'VIP']
for column in binary_features:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])

# Separate the target variable and drop non-predictive columns
y_train = train_data['Transported'].astype(int)
X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])

# Define categorical and numerical features
categorical_features = ['HomePlanet', 'Deck', 'Side', 'Destination']  
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num'] 

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
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

cv_results = grid_search.cv_results_

top5_indices = np.argsort(-cv_results['mean_test_score'])[:5]

# Print out the top 5 scores and their corresponding parameters
print("Top 5 parameter combinations:")
for rank, index in enumerate(top5_indices, start=1):
    print(f"Rank: {rank}")
    print(f"Score: {cv_results['mean_test_score'][index]}")
    print(f"Parameters: {cv_results['params'][index]}\n")