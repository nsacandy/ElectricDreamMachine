import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv("dev.csv")

# Separate features and target variable for training data
X_train = train_data.drop(columns=['Transported'])
y_train = train_data['Transported'].astype(int)

# Impute missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)

# Initialize a list to store accuracy scores for each fold
accuracy_scores = []

# Perform 5-fold cross-validation
for train_index, test_index in skf.split(X_train_scaled, y_train):
    # Split the data into training and testing sets
    X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train_fold, y_train_fold)
    
    # Predict on the test fold
    y_pred = model.predict(X_test_fold)
    
    # Compute accuracy and store it
    accuracy = accuracy_score(y_test_fold, y_pred)
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy across all folds
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print("Average Accuracy:", average_accuracy)

"""
# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform it
X_train_imputed = imputer.fit_transform(X_train)

# Load testing data from CSV
test_data = pd.read_csv("pretest.csv")

# Separate features and target variable for testing data
X_test = test_data.drop(columns=['Transported'])
y_test = test_data['Transported']

# Transform the testing data using the same imputer
X_test_imputed = imputer.transform(X_test)

# Standardize features using scaler fitted on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on testing data
test_predictions = model.predict(X_test_scaled)

# Evaluate model
test_accuracy = model.score(X_test_scaled, y_test)
print("Testing Accuracy:", test_accuracy)
"""
"""
Results: 
Before K-fold validation, we are getting  .7998 accuracy. I assume that number will be smaller in K fold cross-validation.
After K-Fold validation, we have an accuracy rating of .78494
"""
