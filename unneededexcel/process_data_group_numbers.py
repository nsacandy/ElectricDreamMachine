import pandas as pd

data = pd.read_csv("traintestsplit.csv")

# Count the group size and add it to a new column
data['GroupSize'] = data.groupby('GroupId')['GroupId'].transform('size')

# Save the updated dataframe to a new csv file
data.to_csv("traintestsplit_groupcount.csv", index=False)

# Provide the file path to the new CSV
"traintestsplit_groupcount.csv"