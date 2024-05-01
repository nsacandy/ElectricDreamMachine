import pandas as pd

# Load the data
data = pd.read_csv("traintestcombo.csv")

# Split the PassengerId to extract GroupId
data['GroupId'] = data['PassengerId'].apply(lambda x: x.split('_')[0])

# Split the Cabin into Deck, Num, and Side
data[['Deck', 'Num', 'Side']] = data['Cabin'].str.split('/', expand=True)

# Save the new dataframe to a new csv file
data.to_csv("traintestsplit.csv", index=False)