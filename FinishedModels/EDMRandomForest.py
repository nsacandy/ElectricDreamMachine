import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
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

# Create a preprocessing and modeling pipeline with feature selection
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
                           ('classifier', RandomForestClassifier(random_state=3270))])

# Define the parameter grid for Random Forest
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)

# Perform randomized search
random_search_rf = RandomizedSearchCV(pipeline, param_grid_rf, n_iter=100, cv=skf, verbose=2, n_jobs=-1, random_state=3270)
random_search_rf.fit(X_train, y_train)

cv_results = random_search_rf.cv_results_

top5_indices = np.argsort(-cv_results['mean_test_score'])[:5]

# Print out the top 5 scores and their corresponding parameters
print("Top 5 parameter combinations:")
for rank, index in enumerate(top5_indices, start=1):
    print(f"Rank: {rank}")
    print(f"Score: {cv_results['mean_test_score'][index]}")
    print(f"Parameters: {cv_results['params'][index]}\n")
