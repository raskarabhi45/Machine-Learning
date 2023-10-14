# Diabetes Predictor application using Decision Tree Algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

print("----Marvellous Infosystems by Piyush Khairnar-----")

print("------Diabetes predictor using Decision Tree------")

diabetes=pd.read_csv('diabetes.csv')

print("Columns of dataset")
print(diabetes.columns)

print("First 5 records of dataset")
print(diabetes.head())

print("Dimension of diabetes data : {}".format(diabetes.shape))

x_train,x_test,y_train,y_test=train_test_split(diabetes.loc[:,diabetes.columns!='Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

tree=DecisionTreeClassifier(random_state=0)
tree.fit(x_train,y_train)

print("Accuracy of training set : {:.3f}".format(tree.score(x_train,y_train)))
print("Accuracy on test set : {:.3f}".format(tree.score(x_test,y_test)))

tree=DecisionTreeClassifier(max_depth=3,random_state=0)
tree.fit(x_train,y_train)

print("Accuracy of training set : {:.3f}".format(tree.score(x_train,y_train)))
print("Accuracy on test set : {:.3f}".format(tree.score(x_test,y_test)))

print("Feature importance : \n{}".format(tree.feature_importances_))


def plot_feagure_importance_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features=8
    plt.barh(range(n_features),model.feature_importances_,align='center')
    diabetes_features=[x for i, x in enumerate(diabetes.columns)if i!=8]
    plt.yticks(np.arange(n_features),diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)
    plt.show()

plot_feagure_importance_diabetes(tree)

"""Output
----Marvellous Infosystems by Piyush Khairnar-----
------Diabetes predictor using Decision Tree------
Columns of dataset
Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')
First 5 records of dataset
   Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72  ...                     0.627   50        1
1            1       85             66  ...                     0.351   31        0
2            8      183             64  ...                     0.672   32        1
3            1       89             66  ...                     0.167   21        0
4            0      137             40  ...                     2.288   33        1

[5 rows x 9 columns]
Dimension of diabetes data : (768, 9)
Accuracy of training set : 1.000
Accuracy on test set : 0.714
Accuracy of training set : 0.773
Accuracy on test set : 0.740
Feature importance :
[0.04554275 0.6830362  0.         0.         0.         0.27142106
 0.         0.        ]"""
