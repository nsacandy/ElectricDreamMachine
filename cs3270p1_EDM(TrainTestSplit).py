"""
Dataset Splitting Script for Transportation Prediction.

This script splits a preprocessed dataset into development (dev) and test sets for
the purpose of training and evaluating machine learning models in a
transportation prediction task. The script assumes the presence of a
file named 'preprocessed_train_kaggle.csv', which is expected to contain
preprocessed features along with a target
variable named 'Transported'.

The script uses a stratified split to ensure that the distribution of the
target variable is similar in both the development and test sets. This approach
helps in maintaining the balance of classes across the datasets,
which is particularly useful in classification tasks.

Usage:
- Place the 'preprocessed_train_kaggle.csv' file in the same directory as this script.
- Run the script to generate 'dev.csv' and 'test.csv' files, representing the
development and test sets, respectively.

Dependencies:
- pandas: For reading the CSV file and handling data.
- sklearn.model_selection: For splitting the dataset in a stratified manner.

Output:
- 'dev.csv': The development set, containing both features and the target variable.
- 'test.csv': The test set, containing both features and the target variable.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("preprocessed_train_kaggle.csv")

__author__ = 'Thomas Lamont, Nic Sacandy, Dillon Emmons'
__version__ = 'Spring 2024'
__pylint__= '2.14.5'
__pandas__ = '1.4.3'
__numpy__ = '1.23.1'
TARGET = 'Transported'

X_dev, X_test, y_dev, y_test = train_test_split(
    data.drop(columns=TARGET),
    data[TARGET],
    stratify=data[TARGET],
    test_size=0.20,
    random_state=3270
)

dev_set = pd.concat([X_dev, y_dev], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

dev_set.to_csv("dev.csv", index=False)
test_set.to_csv("test.csv", index=False)

print("Development set and test set have been created and saved.")
