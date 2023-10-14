#Using Decision Tree classifier algorithm

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import train_test_split

def MarvellousDecisionTreeClassifier():
     Dataset=load_iris()          #1 Load  the data

     Data=Dataset.data  # 150 records 4 columns
     Target=Dataset.target # 150 records 1 column

    #2=manipulate the data 
     Data_train, Data_test,Target_train,Target_test=train_test_split(Data,Target,test_size=0.5)  

     Classifier=DecisionTreeClassifier()  # class name

    # 3 build the data
     Classifier.fit(Data_train,Target_train)

    # 4 test the data
     Predictions=Classifier.predict(Data_test)

     Accuracy=accuracy_score(Target_test,Predictions)

    # 5 improve =missing

     return Accuracy


def main():
    ret=MarvellousDecisionTreeClassifier()

    print("Accuracy of Iris dataset with KNN is : ",ret*100)


if __name__=="__main__":
    main()