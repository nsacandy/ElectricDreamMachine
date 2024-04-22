import pandas as pd

# Load the dataset
data = pd.read_csv("preprocessed_combo.csv")

# Select the columns to be one-hot encoded
categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# Use get_dummies to one-hot encode the categorical columns
data = pd.get_dummies(data, columns=categorical_columns)

# Save the new dataset with dummy variables to a CSV file
data.to_csv("dummies_combo.csv", index=False)

# Print completion message
print("Dataset with dummy variables saved to 'dummies_combo.csv'")