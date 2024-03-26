import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

train_data = pd.read_csv("dev.csv")

# Separate features and target variable for training data
X_train = train_data.drop(columns=['Transported'])
y_train = train_data['Transported'].astype(int)

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
