import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("preprocessed_train_kaggle.csv")

# Assuming 'Transported' is the target column for classification
target = 'Transported'

# Split the data into development (80%) and test (20%) sets in a stratified manner
X_dev, X_test, y_dev, y_test = train_test_split(
    data.drop(columns=target),  
    data[target],               
    stratify=data[target],      
    test_size=0.20,             
    random_state=3270          
)

# Combine features and target for development and test sets
dev_set = pd.concat([X_dev, y_dev], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# Save the development and test sets into new csv files
dev_set.to_csv("dev.csv", index=False)
test_set.to_csv("test.csv", index=False)

print("Development set and test set have been created and saved.")