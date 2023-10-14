# Breast Cancer Dataset with support Vector Machine'

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


def MarvellousSVM():
    # Load dataset
    cancer=datasets.load_breast_cancer()

    # print the names of the 13 features
    print("Features of the cancer dataset : ",cancer.feature_names)

    # print the label type of cancer ('malignant',benign)
    print("Lables of the cancer dataset : ",cancer.target_names)

    # print data(feature) shape
    print("Shape of dataset is : ",cancer.data.shape)

    # print the cancer data features of top 5 records
    print("First 5 records are : ")
    print(cancer.data[0:5])

    # print the cancer labels (0:miligant , 1: benign)
    print("Target of dataset : ",cancer.target)

    # split dataset into training and testing dataset

    x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.3, random_state=109)

    # Create a svm classifier
    clf=svm.SVC(kernel='linear')  # Linear model

    # Train the model using the training datasets
    clf.fit(x_train,y_train)

    # predict the response for test dataset
    y_pred=clf.predict(x_test)

    # Model accuracy : how often is the classifer correct ?
    print("Accuracy of the model is : ",metrics.accuracy_score(y_test,y_pred)*100)

def main():
    print("---------------Marvellous Support Vector Machine------------")

    MarvellousSVM()

if __name__=="__main__":
    main()