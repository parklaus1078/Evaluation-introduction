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

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, fake_pred))
#      0   1
# 0  405   0
# 1   45   0

# Precision
from sklearn.metrics import precision_score

print('\nPrecision Score :', precision_score(y_test, fake_pred))

# Recall
from sklearn.metrics import recall_score

print('\nRecall Score :', recall_score(y_test, fake_pred))

# Trade-off
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)

cm = confusion_matrix(y_test, pred)
cmIndex = ['Actual_Negative', 'Actual_Positive']
cmCols = ['Predicted_Negative', 'Predicted_Positive']

print("\n### Evaluation of Confusion Matrix, Accuracy, Precision, Recall ###\n")
print("Confusion Matrix : \n", pd.DataFrame(cm, index=cmIndex, columns=cmCols).to_string())
print("\nAccuracy Score :", np.round(accuracy_score(y_test, pred), 4))
print("\nPrecision Score :", np.round(precision_score(y_test, pred), 4))
print("\nRecall Score :", np.round(recall_score(y_test, pred), 4))

pred_proba = lr_clf.predict_proba(X_test)
pred = lr_clf.predict(X_test)
print("\npred_proba() 결과 shape : {0}".format(pred_proba.shape))
print("\npred_proba array에서 앞 5개만 샘플로 추출 :\n", pd.DataFrame(pred_proba, columns=['Probability of rv being 0', 'Probability of rv being 1']).head().to_string())

# Concatenate prediction probability array and predicted value array.
pred_proba_result = np.concatenate([pred_proba, pred.reshape(-1, 1)], axis=1)
print("\nAmong those 2 classes, predict the class value with the one with the higher probability\n", pd.DataFrame(pred_proba_result, columns=["Prob. of Rv being 0", "Prob. of Rv being 1", "Rv Value"]).head().to_string())

# Binarizer
from sklearn.preprocessing import Binarizer
X = [[1,  -1,   2],
     [2,   0,   0],
     [0, 1.1, 1.2]]

# If an element is less than or equal to threshold, return 0. Return 1, otherwise.
binarizer = Binarizer(threshold=1.1)
print(binarizer.fit_transform(X))

# Set Binarizer's Threshold
custom_threshold = 0.5

# Apply Binarizer to the second column of predict_proba
pred_proba_1 = pred_proba[:,1].reshape(-1, 1)
binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba_1)

cm = confusion_matrix(y_test, custom_predict)
print("\n### Evaluation of Confusion Matrix, Accuracy, Precision, Recall ###\n")
print("Confusion Matrix : \n", pd.DataFrame(cm, index=cmIndex, columns=cmCols).to_string())
print("\nAccuracy Score :", np.round(accuracy_score(y_test, custom_predict), 4))
print("\nPrecision Score :", np.round(precision_score(y_test, custom_predict), 4))
print("\nRecall Score :", np.round(recall_score(y_test, custom_predict), 4))


def get_eval_by_threshold(y, pred_prob, thresholds):
    for th in thresholds:
        binarizer = Binarizer(threshold=th).fit(pred_prob)
        pre = binarizer.transform(pred_prob)
        print("Threshold :", th)
        cm = confusion_matrix(y, pre)
        print("\n### Evaluation of Confusion Matrix, Accuracy, Precision, Recall ###\n")
        print("Confusion Matrix : \n", pd.DataFrame(cm, index=cmIndex, columns=cmCols).to_string())
        print("\nAccuracy Score :", np.round(accuracy_score(y, pre), 4))
        print("\nPrecision Score :", np.round(precision_score(y, pre), 4))
        print("\nRecall Score :", np.round(recall_score(y, pre), 4))

test_thresholds = [.35, .4, .45, .5, .55, .6]
test_pred_proba = pred_proba[:,1].reshape(-1, 1)
get_eval_by_threshold(y_test, test_pred_proba, test_thresholds)

# precision_recall_curve()
from sklearn.metrics import precision_recall_curve

pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1)
print("\nThresholds array's shape :", thresholds.shape)
print("Precisions array's shape :", precisions.shape)
print("Recalls array's shape :", recalls.shape)

print("Threshold samples :", thresholds[:5])
print("Precision samples :", precisions[:5])
print("Recall samples :", recalls[:5])

thr_index = np.arange(0, thresholds.shape[0], 15)
print("10 index of thresholds to extract sample :", thr_index)
print("10 sample thresholds :", np.round(thresholds[thr_index], 2))
print("10 sample precisions :", np.round(precisions[thr_index], 3))
print("10 sample recalls :", np.round(recalls[thr_index], 3))

# Draw precision_recall_curve plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def precision_recall_curve_plot(y, pred_prob):
    precisions, recalls, thresholds = precision_recall_curve(y, pred_prob)

    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle="--", label="precision")
    plt.plot(thresholds, recalls[0:threshold_boundary], label="recall")

    # X-axis(threshold)'s scale unit to 0.1
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, .1), 2))

    # Setting X-axis, y-axis, label and legend
    plt.xlabel("Threshold value")
    plt.ylabel("Precision and Recall value")
    plt.legend()
    plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, lr_clf.predict_proba(X_test)[:,1])
