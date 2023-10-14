#Using K naerest neighbour algorithm
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection  import train_test_split

def MarvellousKNeighborsClassifier():
     Dataset=load_iris()          #1 Load  the data

     Data=Dataset.data     # 150 records 4 columns sepal length ,sepal width , petal length ,petal width A and C
     Target=Dataset.target # 150 records 1 column  classes either sentosa,versicolor and verginica B and D

    #2 manipulate the data  
    #data shuffle krun todun dete generally split 70 30 percent for training and testing resp
     Data_train, Data_test,Target_train,Target_test=train_test_split(Data,Target,test_size=0.5)  

    #creation of class object
     Classifier=KNeighborsClassifier()

    # 3 build the data
     Classifier.fit(Data_train,Target_train)

    # 4 test the data
     Predictions=Classifier.predict(Data_test)

     Accuracy=accuracy_score(Target_test,Predictions)

    # 5 improve =missing

     return Accuracy


def main():
    ret=MarvellousKNeighborsClassifier()

    print("Accuracy of Iris dataset with KNN is : ",ret*100)


if __name__=="__main__":
    main()