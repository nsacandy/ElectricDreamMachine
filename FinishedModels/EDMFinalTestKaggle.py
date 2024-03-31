import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Load the development and test data
train_data = pd.read_csv("preprocessed_train.csv")
test_data = pd.read_csv("preprocessed_test.csv")

# Separate features and target variable for development and test data
X_train = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])
y_train = train_data['Transported'].astype(int)

X_test = test_data.drop(columns=['PassengerId', 'Name', 'Cabin'])

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
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001, random_state=3270))])

# Logistic Regression hyperparameters and pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logistic', LogisticRegression(C=100, max_iter=300, solver='liblinear', random_state=3270))])

# Random Forest hyperparameters and pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('random_forest', RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=10, random_state=3270))])

# Fit the pipelines to the training data
nn_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# Preprocess the unknown data and make predictions
unknown_preprocessed = preprocessor.transform(test_data.drop(columns=['PassengerId', 'Name', 'Cabin']))
nn_pred = nn_pipeline.named_steps['mlp'].predict(unknown_preprocessed)
lr_pred = lr_pipeline.named_steps['logistic'].predict(unknown_preprocessed)
rf_pred = rf_pipeline.named_steps['random_forest'].predict(unknown_preprocessed)

# Ensemble prediction: if two or more models predict True, overall prediction is True
ensemble_pred = (nn_pred.astype(int) + lr_pred.astype(int) + rf_pred.astype(int)) >= 2

# Create a DataFrame with just PassengerId and ensemble predictions
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': ensemble_pred})

# Convert boolean predictions to 'TRUE'/'FALSE' string as per your requirements
output['Transported'] = output['Transported'].map({True: 'True', False: 'False'})

# Export to CSV
output.to_csv("RF2Prediction.csv", index=False)

print("Ensemble predictions exported to EnsemblePredictions.csv")