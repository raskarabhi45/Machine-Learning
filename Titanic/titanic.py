"""
There is one data set  which contains information about the passengers from 
titanic.
This data set describe multiple features about survived and non survived passengers

Consider below characteristics of ML Application

classifier : Logistic regression
dataset    : titanic dataset
Features   : Passenger id , Gender, Age, fare , class etc
Labels     : -

Consider below application which uses Logistic Regression algorithm from skit laern \+
library to train above data set and predict whether passenger survived or not"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def MarvellousTitanicLogistic():
    # step 1 : Load data

    titanic_data=pd.read_csv(r"C:\Users\ABHIJEET\Desktop\Python\ML\Titanic\titanic.csv")
    #r"C:\Users\Welcome\Desktop\Sales.csv",encoding='latin1'

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Numbers of passangers are "+str(len(titanic_data)))

    # step 2 : Analys data
    print("Visualisation : Survived and non survived passangers")
    figure()
    target="Survived"

    countplot(data=titanic_data,x=target).set_title("Marvellous Infosystems : Survived and non survived passangers")
    show()

    print("Visualiastion : Survived and non survived passangres based on Gender")
    figure()
    target="Survived"

    countplot(data=titanic_data,x=target,hue="Sex").set_title("Marvellous Infosystems : Survived and non survived passangers based on gender")
    show()

    print("Visualiastion : Survived and non survived passangres based on class")
    figure()
    target="Survived"

    countplot(data=titanic_data,x=target,hue="PClass").set_title("Marvellous Infosystems : Survived and non survived passangers based on class")
    show()


    print("Visualiastion : Survived and non survived passangres based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Marvellous Infosystems : Survived and non survived passangers based on Age")
    show()

    print("Visualisation : Survied and non survied passangers based on the fire")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Marvellous Infosystems : Survived and non survived passangers based on Fare")
    show()

    # step 3 : data Cleaning
    titanic_data.drop("zero",axis=1,inplace=True)

    print("First 5 entries loaded dataset after removing zero column")
    print(titanic_data.head(5))

    print("Values of sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of sex column after removing one field")
    Sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
    print(Sex.head(5))

    print("Values of Pclass column after removing one field")
    PClass=pd.get_dummies(titanic_data["PClass"],drop_first=True)
    print(PClass.head(5))

    print("Values of dataset after concenating new columns")
    titanic_data=pd.concat([titanic_data,Sex,PClass],axis=1)
    print(titanic_data.head(5))

    print("Values of data set after removing irelevvant columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head(5))

    x=titanic_data.drop("Survived",axis=1)
    y=titanic_data["Survived"]

    # step  4 : data Training
    xtrain, xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)

    logmodel=LogisticRegression()

    logmodel.fit(xtrain,ytrain)

    # step 4 : data testing
    prediction=logmodel.predict(xtest)

    # step 5 : calculate accuracy
    print("Classification report of Logistic Regression is :")
    print(classification_report(ytest,prediction))

    print("Confusion matrix of Logistic Regression is : ")
    print(confusion_matrix(ytest,prediction))

    print("Accuracy of Logistic Regression is : ")
    print(accuracy_score(ytest,prediction))




def main():
    print("------Marvellous Infosystems by piyush Khairnar------")

    print("Supervised Machine Learning")

    print("Logistic Regression on Titanic data set")

    MarvellousTitanicLogistic()


if __name__=="__main__":
    main()