import pandas as pd

# Load the dataset
data = pd.read_csv("dummies_test.csv")  # Replace with your actual file path

# Calculate the median of the 'Num' column, excluding NaNs
median_num = data['Num'].median()

# Fill NaN values in 'Num' column with the median
data['Num'] = data['Num'].fillna(median_num)

# Save the updated dataset to a new CSV file
data.to_csv("dummies_test.csv", index=False)  # Replace with your desired output file path

# Print completion message
print("Missing 'Num' values have been filled with the median.")