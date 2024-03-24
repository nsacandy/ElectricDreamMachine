import pandas as pd
from sklearn.model_selection import StratifiedKFold as skfold

train_data_ = pd.read_csv("train.csv")

features = ["HomePlanet","CryoSleep","Age","Destination","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]

X = pd.get_dummies(train_data_[features])
labels = train_data_["Transported"]

skf = skfold(n_splits=5, shuffle=True, random_state=3270)

for train_index, test_index in skf.split(X, labels):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    labels_train, labels_test = labels.iloc[train_index], labels.iloc[test_index]
    
    # Now you can use X_train, X_test, labels_train, and labels_test for training and testing
    
X_train.to_csv("dev.csv")
X_test.to_csv("pretest.csv")
