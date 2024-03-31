import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

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

    # Initialize MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001,
                    solver='adam', verbose=10, random_state=42, tol=0.0001)
    # Train the classifier
    mlp.fit(X_train_fold, y_train_fold)

    # Predict labels for the test set
    y_pred = mlp.predict(X_test_fold)

    # Compute accuracy and store it
    accuracy = accuracy_score(y_test_fold, y_pred)
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy across all folds
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print("Average Accuracy:", average_accuracy)

    
