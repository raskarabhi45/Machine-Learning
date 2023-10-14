#Clustering using k - mean algorithm
"""
In this case study we are generating the data set at run time randomly and apply 
user defined k mean algorithm"""

import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt

def MarvellousKMean():
    print("_________________________")
    # set three centers , the model should predict similar results
    center_1=np.array([1,1])
    print(center_1)
    print("_________________________")

    center_2=np.array([5,5])
    print(center_2)
    print("_________________________")

    center_3=np.array([8,1])
    print(center_3)
    print("_________________________")

    # Generate random data and center it to the three centers
    data_1=np.random.randn(7,2) + center_1
    print("Elements of first cluster with size "+str(len(data_1)))
    print(data_1)
    print("_________________________")

    data_2=np.random.randn(7,2) + center_2
    print("Elements of second cluster with size "+str(len(data_2)))
    print(data_2)
    print("_________________________")

    data_3=np.random.randn(7,2) + center_3
    print("Elements of first cluster with size "+str(len(data_3)))
    print(data_3)
    print("_________________________")

    data=np.concatenate((data_1,data_2,data_3),axis=0)
    print("Size of complete dataset "+str(len(data)))
    print(data)
    print("_________________________")

    plt.scatter(data[:,0],data[:,1],s=7)
    plt.title('Marvellous Infosystems : Input dataset')
    plt.show()
    print("_________________________")

    # Numbers of clusters
    k=3
    # Number of training data
    n=data.shape[0]
    print("total number of element are ",n)
    print("_________________________")

    # number of features in the data
    c=data.shape[1]
    print("Total number of features are ",c)
    print("_________________________")
    
    # generate random centers , hee we use sigma and mean to ensure it represent the whole data
    mean=np.mean(data,axis=0)
    print("value of mean ",mean)
    print("_________________________")

    # calculate standard deviation
    std=np.std(data,axis=0)
    print("value of std ",std)
    print("_________________________")

    centers=np.random.randn(k,c)*std+mean
    print("Random points are ",centers)
    print("_________________________")

    #plot the data and the centers generated as random
    plt.scatter(data[:,0],data[:,1],c='r',s=150)
    plt.scatter(centers[:,0],centers[:,1],marker='x',c='g',s=150)
    plt.title('Marvellous infosystems : Imput Database with random centroid *')
    plt.show()
    print("_________________________")

    centers_old=np.zeros(centers.shape) # to store old centers
    centers_new=deepcopy(centers) # to store new centers

    print("Values of old centroids")
    print(centers_old)
    print("_________________________")

    print("Values of new centroids")
    print(centers_new)
    print("_________________________")

    data.shape
    clusters=np.zeros(n)
    distances=np.zeros((n,k))

    print("Initial distances are ")
    print(distances)
    print("_________________________")

    error=np.linalg.norm(centers_new-centers_old)
    print("Values of error is ",error);
    # When  ,after an update the estimate of that center stays the same , exit loop

    while error!=0:
        print("Value of error is ",error)
        # measure the distance to every center
        print("measure the distance to every center")
        for i in range(k):
            print("Iteration number ",i)
            distances[:,i]=np.linalg.norm(data-centers[i],axis=1)

        # assign all traiing data to closest center
        clusters=np.argmin(distances,axis=1)

        centers_old=deepcopy(centers_new)

        # calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i]=np.mean(data[clusters==i],axis=0)
        error=np.linalg.norm(centers_new-centers_old)

        # end of while
        centers_new

        # plot the data and the centers generated as random
        plt.scatter(data[:,0],data[:,1],s=7)
        plt.scatter(centers_new[:,0],centers_new[:,1],marker='*',c='g',s=150)
        plt.title('Marvellous Infosystems : Final data with centroid')
        plt.show()




def main():
    print("------Marvellous Infosystems by Piyush Khairnar-----")

    print("Unsupervised Machine learning")

    print("Clustering using k mean algorithm")

    MarvellousKMean()


if __name__=="__main__":
    main()