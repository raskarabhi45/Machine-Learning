# Diabetes Predictor application using Logistic Regression Algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)

print("----Marvellous Infosystems by Piyush Khairnar-----")

print("------Diabetes predictor using Logostic Regression------")

diabetes=pd.read_csv('diabetes.csv')

print("Columns of dataset")
print(diabetes.columns)

print("First 5 records of dataset")
print(diabetes.head())

print("Dimension of diabetes data : {}".format(diabetes.shape))

x_train,x_test,y_train,y_test=train_test_split(diabetes.loc[:,diabetes.columns!='Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

logreg=LogisticRegression().fit(x_train,y_train)

print("Accuracy of training set : {:.3f}".format(logreg.score(x_train,y_train)))
print("Accuracy on test set : {:.3f}".format(logreg.score(x_test,y_test)))

logreg001=LogisticRegression(C=0.01).fit(x_train,y_train)

print("Accuracy of training set : {:.3f}".format(logreg001.score(x_train,y_train)))
print("Accuracy on test set : {:.3f}".format(logreg001.score(x_test,y_test)))

