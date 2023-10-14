# Diabetes Predictor application using K Nearest Neighbours

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


print("----Marvellous Infosystems by Piyush Khairnar-----")

print("------Diabetes predictor using K Nearest neighbour------")

diabetes=pd.read_csv('diabetes.csv')

print("Columns of dataset")
print(diabetes.columns)

print("First 5 records of dataset")
print(diabetes.head())

print("Dimension of diabetes data : {}".format(diabetes.shape))

x_train,x_test,y_train,y_test=train_test_split(diabetes.loc[:,diabetes.columns!='Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

training_accuracy=[]
testing_accuracy=[]

# try n_neighbour from 1 to 10
neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    # build the model
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train,y_train)
    #record training set accuracy
    training_accuracy.append(knn.score(x_train,y_train))
    # record test set accuracy
    testing_accuracy.append(knn.score(x_test,y_test))

plt.plot(neighbors_settings,training_accuracy,label="training accuracy")
plt.plot(neighbors_settings,testing_accuracy,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighnors")
plt.legend()
plt.savefig('knn_compare_model')
plt.show()

knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)

print("Accuracy of the knn on training set : {:.2f}".format(knn.score(x_train,y_train)))


print("Accuracy of the knn on testing set : {:.2f}".format(knn.score(x_test,y_test)))

""" Output
----Marvellous Infosystems by Piyush Khairnar-----
------Diabetes predictor using K Nearest neighbour------
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
Accuracy of the knn on training set : 0.79
Accuracy of the knn on testing set : 0.78
"""