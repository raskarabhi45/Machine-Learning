# Breast Cancer Case Study with6 Insustial Programmimg Approach

###########################################
# Required Python packages
###########################################

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

###########################################
# File paths
###########################################
INPUT_PATH="breast-cancer-wisconsin.data"
OUTPUT_PATH="breast-cancer-wisconsin.csv"

###########################################
#Headers
###########################################

HEADERS=["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","Mitoses","CancerType"]


###########################################
# Function name : read_data
# Description : Read the data into pandas dataframe
# input : path nof csv file
# output : Gives the data
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################

def read_data(path):
    data=pd.read_csv(path)
    return data


###########################################
# Function name : get_headers
# Description : dataset headers
# input :  dataset
# output : Returns the header
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################

def get_headers(dataset):
    return dataset.columns.values

###########################################
# Function name : add_headers
# Description : add the headers to the dataset
# input :  dataset
# output : Updated dataset
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################

def add_headers(dataset,headers):
    dataset.columns=headers
    return dataset

###########################################
# Function name : data_file_to_csv
# input : Nothing
# output : write the data to csv
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################


def data_file_to_csv():
    #Headers
    headers=["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","Mitoses","CancerType"]
    # Load the dataset into pandas dataframe
    dataset=read_data(INPUT_PATH)
    # add the headers to the loaded dataset
    dataset=add_headers(dataset,headers)
    # save the loaded dataset into the csv format
    dataset.to_csv(OUTPUT_PATH,index=False)
    print("File Saved... !")

###########################################
# Function name : split_dataset
# Description : split the dataset with train percentage
# input :  dataset with related information
# output : dataset after splitting
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################

def split_dataset(dataset,train_percentage,feature_headers,target_headers):

    # split the dataset into train and test dataset

    train_x,test_x,train_y,test_y=train_test_split(dataset[feature_headers],dataset[target_headers],train_size=train_percentage)
    return train_x,test_x,train_y,test_y

###########################################
# Function name : handle missing values
# Description : filter missing values from dataset
# input :  dataset with missing values
# output : dataset by removing missing values
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################

def handel_missing_values(dataset,missing_values_header,missing_label):
    return dataset[dataset[missing_values_header]!=missing_label]


###########################################
# Function name : random forest classifier
# Description : to train the random forest classifier with features and target data
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################

def random_forest_classifier(features,target):
    clf=RandomForestClassifier()
    clf.fit(features,target)
    return clf


###########################################
# Function name : dataset_statistics
# Description : basic statistics of the dataset
# input :  dataset 
# output : description of dataset
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################

def dataset_statistics(dataset):
    print(dataset.describe())


###########################################
# Function name : main
# Description : main function from where execution starts
# Author : Abhishek Narendra Raskar
# Date : 22/01/2023
###########################################

def main():
    # Load the csv file into pandas dataframe
    dataset=pd.read_csv(OUTPUT_PATH)
    #get basic statistics of the loaded dataset
    dataset_statistics(dataset)

    # filter missing values
    dataset=handel_missing_values(dataset,HEADERS[6],'?')

    train_x,test_x,train_y,test_y=split_dataset(dataset,0.7,HEADERS[1:-1],HEADERS[-1])

    # Train and test dataset size details
    print("train_x Shape :: ",train_x.shape)
    print("train_y Shape :: ",train_y.shape)
    print("test_x Shape :: ",test_x.shape)
    print("test_y Shape :: ",test_y.shape)

    # Create random forest classifier instance
    trained_model=random_forest_classifier(train_x,train_y)
    print("Trained Mode :: l",trained_model)
    predictions=trained_model.predict(test_x)

    for i in range(0,205):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i],predictions[i]))

    print("Train Accuracy :: ", accuracy_score(train_y,trained_model.predict(train_x)))
    print("Test Accuracy ::",accuracy_score(test_y,predictions))
    print("Confusion Matrix ",confusion_matrix(test_y,predictions))



###########################################
# Application Starter
###########################################

if __name__=="__main__":
    main()



  