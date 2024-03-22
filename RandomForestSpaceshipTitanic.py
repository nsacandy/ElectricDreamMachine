
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
 
#Turning data into pandas dataframe
train_data_ = pd.read_csv("../../.kaggle/train.csv")

test_data_ = pd.read_csv("../../.kaggle/test.csv")

labels = train_data_["Transported"]

#Currently analyzes these features/words
features = ["HomePlanet","CryoSleep","Age","Destination","VIP","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]

X = pd.get_dummies(train_data_[features])
X_test = pd.get_dummies(test_data_[features])

hypothesis = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)

hypothesis.fit(X,labels)
predictions = hypothesis.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data_.PassengerId, 'Transported': predictions})

#Writes the submission file based on the sample submission file     
output.to_csv("fake_submission.csv",index=False)

#Just a quick diagnostic print statement
print (output.tail(5))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
