#There is one data which classify the wines according to its contents into three classes
"""
These data are the results of chemical analysis of wines grown in the same region in italy but derived from
three culticars. The analysis determined the quatities of 13 constutents found in each of three types of wines\
    
wine dataset contains 13 features
1) Alcolhol
2) Malic Acid
3) Ash
4) Alcalinity of ash
5) Magnesium
6) Total phenols
7) Flavonoids
8) Nonflavanoid phenols
9) Proanthocyanins
10)color intensity
11)Hue
12)OD280/OD315 of diluted wines
13)Proline

class 1  class 2 class 3
"""

"""
Classifier : K Nearest Neighbour
Dataset    : Wine Predictor dataset
Features   :---
labels     : class 1 , class 2 ,class 3
Training Dataset : 70% of 178 entries
testing dataset  : 30% of 178 entries

Consider below Machine Learning Application"""


from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def WinePredictor():
    #Load dataset
    wine=datasets.load_wine()

    #Print the names of the features
    print(wine.feature_names)

    #print the labels species(class_0,class_1,class_2)
    print(wine.target_names)

    #print the wine data (top 5 records)
    print(wine.data[0:5])

    # print the wine labels ( )
    print(wine.target)

    #Split the dataset into trainiing set and testing set
    # 70% training and 30% test
    X_train,X_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)

    #Create KNN Classifier
    knn=KNeighborsClassifier(n_neighbors=3)

    #Train the model using the training sets
    knn.fit(X_train,y_train)

    # predict the responce for test dataset
    y_pred=knn.predict(X_test)

    # Model accuracy , how often is the classifier correct ?
    print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))


def main():
    print("----------Marvellous Infosystems by Piyush Khairnar----------")

    print("Machine Learning Application")

    print("Wine predictor appllication using K Nearest Kneighbor algorithm")

    WinePredictor()


if __name__=="__main__":
    main()