# Using KNeighbors Classifier algorithm
# and Using Decision Tree Classifier

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def MarvellousCalculateAccuracyDecisionTree():
    iris=load_iris()

    data=iris.data
    target=iris.target

    data_train, data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)

    classifier=tree.DecisionTreeClassifier()

    classifier.fit(data_train,target_train)

    predictions=classifier.predict(data_test)

    accuracy=accuracy_score(target_test,predictions)

    return accuracy


def MarvellousCalculateAccuracyKNeighbor():
    iris=load_iris()

    data=iris.data
    target=iris.target

    data_train,data_test,target_train,target_test=train_test_split(data,target,test_size=0.5)

    classifier=KNeighborsClassifier()

    classifier.fit(data_train,target_train)

    predictions=classifier.predict(data_test)

    accuracy=accuracy_score(target_test,predictions)

    return accuracy


def main():
    accuracy=MarvellousCalculateAccuracyDecisionTree()
    print("Accuracy of classification algorithm with Decision Tree Classifer is : ",accuracy*100,"%")

    accuracy=MarvellousCalculateAccuracyKNeighbor()
    print("Accuracy of Classification algorithm with K neighbor classifier is : ",accuracy*100,"%")


if __name__=="__main__":
    main()


