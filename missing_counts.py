import pandas as pd

# Read the data
data = pd.read_csv("preprocessed_combo.csv")

# Count missing values for each column
missing_counts = data.isnull().sum()

# Print the counts
print("Count of missing values for each category:")
print(missing_counts)