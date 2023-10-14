#Case study 3 Play predictor
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def MarvellousPlayPredictor(data_path):
    # step 1 : Load_data
    data=pd.read_csv(data_path,index_col=0)

    print("Size of actual dataset",len(data))

    # step 2 : Clean ,prepare and manipulate the data
    feature_names=['Whether','Temperature']

    print("Names of Features",feature_names)

    whether=data.whether
    Temperature=data.Temperature
    play=data.play

    #Creating labelEncoder
    le=preprocessing.LabelEncoder()

    # converting string labels into numbers
    weather_encoded=le.fit_transform(whether)
    print(weather_encoded)

    temp_encoded=le.fit_transform(Temperature)
    lebel=le.fit_transform(play)

    print(temp_encoded)

    # combining weather and temp into single listof tuples
    features=list(zip(weather_encoded,temp_encoded))

    # step 3 : Train data
    model=KNeighborsClassifier(n_neighbors=3)

    #Train the model using training sets
    model.fit(features,lebel)

    # step 4 : test data
    predicted=model.predict([[0,2]]) # 0:Overcast, 2: Mild
    print(predicted)




def main():
    print("Marvellous Infosystems by Piyush Khairnar")
    print("Machine Learning Application")
    print("Play Predictor application using K Nearest Neighbor Algorithm")

    MarvellousPlayPredictor("PlayPredicor.csv")


if __name__=="__main__":
    main()