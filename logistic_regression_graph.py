import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_csv("dummies_train.csv")
y = train_data['Transported'].astype(int)
X = train_data.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3270)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(C=1, max_iter=300, solver='lbfgs', random_state=3270)
model.fit(X_train_scaled, y_train)

# Get the coefficients of the model
feature_importance = np.abs(model.coef_[0])

# Create a DataFrame for easy plotting
features = pd.DataFrame()
features['Feature'] = X_train.columns
features['Importance'] = feature_importance
features.sort_values(by=['Importance'], ascending=True, inplace=True)

# Plotting
features.set_index('Feature', inplace=True)
features['Importance'].plot(kind='barh', figsize=(10, 8))
plt.title('Feature Importance')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)
plt.show()