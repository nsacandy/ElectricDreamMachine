import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("train.csv")

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
cabin_features = process_cabin(train_data)

# Combine the cabin data back into the original dataframe
train_data = pd.concat([train_data, cabin_features], axis=1)

# Now drop the original 'Cabin' column
train_data = train_data.drop(columns=['Cabin'])

# Assuming 'Transported' is the target and it's already boolean or binary encoded
y_train = train_data['Transported'].astype(int)
X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name'])  # Assuming 'Name' is non-predictive

# Update categorical and numerical features to include the new 'Cabin' features
categorical_features = ['HomePlanet', 'Deck', 'Side', 'Destination']  # 'Deck' and 'Side' are new
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']  # 'Num' is new

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

# Get the best score and parameters
best_score = random_search_rf.best_score_
best_params = random_search_rf.best_params_

# Print the best score and parameters
print(f"Best score: {best_score}")
print(f"Best parameters: {best_params}")
