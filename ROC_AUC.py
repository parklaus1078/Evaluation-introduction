import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Binarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

titanic_df = pd.read_csv("../../python-project/PerfectGuide/1장/titanic/train.csv")
y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop("Survived", axis=1)

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

### <-------------------------------------------------------------------------------- PreProcessing --------------------------------------------------------------------------------> ###
na_dict = {
    'Age': X_titanic_df['Age'].mean(),
    'Cabin': 'N',
    'Embarked': 'N',
    'Fare': 0
}

drop_cols = ['PassengerId', 'Name', 'Ticket']

form_dict = {
    'Cabin': X_titanic_df['Cabin'].str[:1]
}

feats = ['Cabin', 'Sex', 'Embarked']

X_titanic_df = handle_nas(X_titanic_df, na_dict)
X_titanic_df = drop_features(X_titanic_df, drop_cols)
X_titanic_df = format_features(X_titanic_df, form_dict, feats)
### <-------------------------------------------------------------------------------- /PreProcessing --------------------------------------------------------------------------------> ###
### <-------------------------------------------------------------------------------- Train, Predict --------------------------------------------------------------------------------> ###
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)
lr_clf = LogisticRegression(solver="liblinear")
lr_clf.fit(X_train, y_train)
pred_proba = lr_clf.predict_proba(X_test)
pred = lr_clf.predict(X_test)
### <-------------------------------------------------------------------------------- /Train, Predict --------------------------------------------------------------------------------> ###
### <----------------------------------------------------------------------------------- F1 Score ------------------------------------------------------------------------------------> ###
f1 = f1_score(y_test, pred)
print("F1 score : {0:.4f}".format(f1))


def get_clf_eval(y_test, pred):
    print("\n### Evaluation Indexes ###\n")
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print("Confusion Matrix :\n", confusion)
    print("Accuracy : {0:.4f}".format(accuracy) +
          "\nPrecision : {0:.4f}".format(precision) +
          "\nrecall : {0:.4f}".format(recall) +
          "\nf1 : {0:.4f}".format(f1))


def get_clf_eval_by_threshold(y_test, pred_proba, thresholds):
    for th in thresholds:
        binarizer = Binarizer(threshold=th).fit(pred_proba)
        pred = binarizer.transform(pred_proba)
        get_clf_eval(y_test, pred)


thresholds = [.4, .45, .5, .55, .6]
pred_proba = lr_clf.predict_proba(X_test)
get_clf_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1, 1), thresholds)

pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]
fprs, tprs, thresholds = roc_curve(y_test, pred_proba_class1)
thr_index = np.arange(1, thresholds.shape[0], 5)                    #  Create an array of numbers from 1 to legnth of thresholds by 5 step
print("\nIndex of thresholds array to extract sample :", thr_index)
print("Thresholds extracted by the index :", np.round(thresholds[thr_index], 2))
print("FPR per threshold :", np.round(fprs[thr_index], 3))
print("TPR per threshold :", np.round(tprs[thr_index], 3))


def roc_curve_plot(y_test, pred_proba_c1):
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)

    # Plot ROC curve with fprs and tprs
    plt.plot(fprs, tprs, label='ROC')
    # Plot a diagonal line the diagram
    plt.plot([0,1], [0,1], "k--", label="Random")
    # Change scale unit of FPR's X-axis to 0.1, and X, Y axis setting
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("FPR( 1 - Sensitivity )")
    plt.ylabel("TPR( Recall )")
    plt.legend()
    plt.show()


roc_curve_plot(y_test, pred_proba_class1)

pred_proba = lr_clf.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_proba)
print("ROC AUC score : {0:.4f}".format(roc_score))


def get_clf_eval(y_test, pred=None, pred_proba=None):
    print("\n### Evaluation Indexes ###\n")
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_proba)
    print("Confusion Matrix :\n", confusion)
    print("Accuracy : {0:.4f}".format(accuracy) +
          "\nPrecision : {0:.4f}".format(precision) +
          "\nrecall : {0:.4f}".format(recall) +
          "\nf1 : {0:.4f}".format(f1) +
          "\nroc auc : {0:.4f}".format(roc))