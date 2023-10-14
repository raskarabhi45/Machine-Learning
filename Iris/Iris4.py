# Iris case study

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load data
iris=datasets.load_iris()
x=iris.data
y=iris.target

# Split dataset into training set and testing set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)  # 70 percent and 30%

#Create adaboost classifier object
abc=AdaBoostClassifier(n_estimators=50,learning_rate=1)

#Train adaboost classifer
model=abc.fit(x_train,y_train)

# predict the response for test dataset
y_pred=model.predict(x_test)

print("Acccuracy : ",metrics.accuracy_score(y_test,y_pred))

"""
Output
Acccuracy :  0.9777777777777777"""