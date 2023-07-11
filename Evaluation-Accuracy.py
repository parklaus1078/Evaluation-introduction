import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Accuracy Problem 1
# Define a custom classifier
class MyDummyClassifier(BaseEstimator):
    # fit() method does not learn anything.
    def fit(self, X, y=None):
        pass

    # predict() method predicts 0 if Sex feature is 1, 1 otherwise.
    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if X['Sex'].iloc[i] == 1:   # Is Sex value male?
                pred[i] = 0
            else:                       # or not?
                pred[i] = 1

        return pred


# Preprocessing method 1 : Handles falsy values
def handle_nas(df, cols_dict):
    for col_item in cols_dict.items():
        k, v = col_item
        df[k].fillna(v, inplace=True)

    return df


# Preprocessing method 2 : Drops unnecessary features
def drop_features(df, cols):
    df.drop(cols, axis=1, inplace=True)

    return df


# Preprocessing method 3 : Executes Label Encoding
def format_features(df, formats_dict, features):
    for format_item in formats_dict.items():
        k, v = format_item
        df[k] = v

    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])

    return df

### BEGINNING OF INTRODUCTION TO ACCURACY ###
titanic_df = pd.read_csv("../../python-project/PerfectGuide/1장/titanic/train.csv")
print('\n### Titanic Train Set ###\n', titanic_df.head().to_string())
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1)
print('\n### y values from Titanic Train Set ###\n', y_titanic_df.head().to_string())
print('\n### X values from Titanic Train Set ###\n', X_titanic_df.head().to_string())

na_dict = {
    'Age': X_titanic_df['Age'].mean(),
    'Cabin': 'N',
    'Embarked': 'N',
    'Fare': 0
}
X_titanic_df = handle_nas(X_titanic_df, na_dict)
print('\nHandled Falsy Values!!!')

drop_cols = ['PassengerId', 'Name', 'Ticket']
X_titanic_df = drop_features(X_titanic_df, drop_cols)
print('Dropped features!!!')

form_dict = {
    'Cabin': X_titanic_df['Cabin'].str[:1]
}
feats = ['Cabin', 'Sex', 'Embarked']
X_titanic_df = format_features(X_titanic_df, form_dict, feats)
print('Formatted features!!!')
print('\n### X values from Titanic Train Set after Preprocessing data ###\n', X_titanic_df.head().to_string())

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=0)

custom_clf = MyDummyClassifier()
custom_clf.fit(X_train, y_train)
predicts = custom_clf.predict(X_test)
print("\nDummy Classifier's Accuracy : {0:.4f}".format(accuracy_score(y_test, predicts)))

### Accuracy Problem 2
from sklearn.datasets import load_digits

# Define another custom classifier
class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass


    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# Load MNIST data using load_digits()
digits = load_digits()
print('\nMNIST data : \n', digits.data, '\n\nshape :', digits.data.shape)
print('\nMNIST data Target: \n', digits.target, '\n\nshape :', digits.target.shape)

# If the number of digits is 7, return True, transform it to 1 using astype(int). Otherwise, False is returned.
y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)

# Check imbalanced label data distribution
print("\ny_test's shape :", y_test.shape)
print('Distribution of 0 and 1 in y_test \n', pd.Series(y_test).value_counts())

fake_clf = MyFakeClassifier()
fake_clf.fit(X_train, y_train)
fake_pred = fake_clf.predict(X_test)
print("\nAccuracy when every prediction is 0 : {:.3f}".format(accuracy_score(y_test, fake_pred)))
