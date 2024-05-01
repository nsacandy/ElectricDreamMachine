import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("traintestsplit_groupcount.csv")

# Function to infer CryoSleep status based on spending
def infer_cryosleep(row):
    if pd.isna(row['CryoSleep']):
        if row[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum() == 0:
            return True
        else:
            return False
    else:
        return row['CryoSleep']

# Apply the function to infer CryoSleep
data['CryoSleep'] = data.apply(infer_cryosleep, axis=1)

# Replace missing spending values based on CryoSleep status
spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for column in spending_columns:
    data.loc[data['CryoSleep'] == True, column] = data.loc[data['CryoSleep'] == True, column].fillna(0)
    median_value = data[data['CryoSleep'] == False][column].replace(0, np.nan).median()
    data.loc[(data[column].isna()) & (data['CryoSleep'] == False), column] = median_value

# Get the mode and median values for filling missing data
mode_values = {
    'HomePlanet': data['HomePlanet'].mode()[0],
    'Destination': data['Destination'].mode()[0],
    'VIP': data['VIP'].mode()[0],
    'Deck': data['Deck'].mode()[0],
    'Side': data['Side'].mode()[0]
}
median_values = {
    'Age': data['Age'].median(),
    'Num': data['Num'].median()
}

# Create a group data DataFrame
group_data = data.groupby('GroupId').agg(lambda x: x.mode()[0] if not x.empty and not x.mode().empty else np.nan)

# Function to fill missing values based on group
def fill_missing_with_group_data(row, column, group_data):
    if pd.isna(row[column]):
        if row['GroupId'] in group_data and not pd.isna(group_data.loc[row['GroupId'], column]):
            return group_data.loc[row['GroupId'], column]
        elif column in mode_values:
            return mode_values[column]
        elif column in median_values:
            return median_values[column]
    return row[column]

# Fill missing values based on group data or mode/median
for column in ['HomePlanet', 'Destination', 'VIP', 'Deck', 'Side', 'Age']:
    data[column] = data.apply(lambda row: fill_missing_with_group_data(row, column, group_data), axis=1)

# Ensure that no other values have been mistakenly filled
data['CryoSleep'] = data['CryoSleep'].astype(bool)

# Save the new dataset to a CSV file
data.to_csv("preprocessed_combo.csv", index=False)

# Print completion message
print("Dataset preprocessing complete and saved to 'preprocessed_combo.csv'")
