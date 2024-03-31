import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("preprocessed_train.csv")
test_data = pd.read_csv("preprocessed_test.csv")

# Assuming 'Transported' is the target and it's already boolean or binary encoded
y_train = train_data['Transported'].astype(int)
X_train = train_data.drop(columns=['PassengerId', 'Transported', 'Name', 'Cabin'])  # Assuming 'Name' is non-predictive

# Prepare the test set, aligning it to the training set columns
X_test = test_data.drop(columns=['Name', 'PassengerId', 'Cabin'])

# Update categorical and numerical features to include the new 'Cabin' features
categorical_features = ['HomePlanet', 'Deck', 'Side', 'Destination'] 
numerical_features = ['GroupId', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']  # 'Num' is new

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Using median to handle numerical NaNs
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps into a single transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)]
)

# Initialize the Random Forest Classifier with the best parameters found
best_rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=3270
)

# Create a preprocessing and modeling pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', best_rf_model)])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Preprocess the test data and make predictions
predicted_transport_status = pipeline.predict(X_test)

for i, pred in enumerate(predicted_transport_status):
    print(f"PassengerID: {test_data['PassengerId'].iloc[i]}, Predicted Transported: {'TRUE' if pred else 'FALSE'}")
    
# Create a DataFrame with just PassengerId and predictions
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': predicted_transport_status})

# Export to CSV
output.to_csv("RF2Prediction.csv", index=False)

print("Predictions exported to RF2Prediction.csv")