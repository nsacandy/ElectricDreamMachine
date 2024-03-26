import pandas as pd
from sklearn.model_selection import StratifiedKFold

train_data_ = pd.read_csv("train.csv")

features = ["HomePlanet","CryoSleep","Age","Destination","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]

X = pd.get_dummies(train_data_[features])
y = train_data_["Transported"]

# Concatenate "transported" column with features DataFrame
X_with_transport = pd.concat([X, y], axis=1)

# Assume X is your feature matrix and y is your target variable

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3270)

for train_index, test_index in skf.split(X_with_transport, y):
    X_train, X_test = X_with_transport.iloc[train_index], X_with_transport.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Write to CSV with "transported" column as the last column
X_train.to_csv("dev.csv", index=False)
X_test.to_csv("pretest.csv", index=False)

print (y_train)    

