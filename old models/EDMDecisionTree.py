import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("preprocessed_train.csv")

# Assuming 'Transported' is the target and it's already boolean or binary encoded
y_train = train_data['Transported'].astype(int)
X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name'])  

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
                           ('classifier', DecisionTreeClassifier(random_state=3270))])

# Define the parameter grid for Decision Tree
param_grid_dt = {
    'classifier__max_depth': [None, 10, 20, 30, 40],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4, 6],
    'classifier__max_features': [None, 'sqrt', 'log2'],
    'classifier__criterion': ['gini', 'entropy']
}

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)

# Perform grid search
grid_search_dt = GridSearchCV(pipeline, param_grid_dt, cv=skf, verbose=1, n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

# Get the best score and parameters
best_score = grid_search_dt.best_score_
best_params = grid_search_dt.best_params_

# Print the best score and parameters
print(f"Best score: {best_score}")
print(f"Best parameters: {best_params}")
