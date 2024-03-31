import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the data
train_data = pd.read_csv("dev.csv")

# Assuming 'Transported' is the target and it's already boolean or binary encoded
y_train = train_data['Transported'].astype(int)
X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])  

# Update categorical and numerical features to include the new 'Cabin' features
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
                           ('classifier', LogisticRegression(random_state=3270))])

# Define the parameter grid for Logistic Regression
param_grid_lr = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'classifier__max_iter': [100, 200, 300]
}

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)

# Perform grid search
grid_search_lr = GridSearchCV(pipeline, param_grid_lr, cv=skf, verbose=1, n_jobs=-1, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)

cv_results = grid_search_lr.cv_results_

top5_indices = np.argsort(-cv_results['mean_test_score'])[:5]

# Print out the top 5 scores and their corresponding parameters
print("Top 5 parameter combinations:")
for rank, index in enumerate(top5_indices, start=1):
    print(f"Rank: {rank}")
    print(f"Score: {cv_results['mean_test_score'][index]}")
    print(f"Parameters: {cv_results['params'][index]}\n")