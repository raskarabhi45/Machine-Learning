# Clustering using K-mean algorithm
"""
In this case study we are using Iris dataset with K-mean algorithm from sklearn"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the Iris dataset with pandas
dataset=pd.read_csv('iris.csv')
x=dataset.iloc[:,[0,1,2,3]].values
# finding the optimum number of clusters for  k-means classification
from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the result onto a line graph , allowing us to oberve 'The elbow'
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('ECSS') # within cluster sum of squares
plt.show()

# Applying kmeans to the dataset / creating the kmeans classifier
kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Iris-sentosa')

plt.scatter(x[y_kmeans==1,0],x[y_kmeans==0,1],s=100,c='blue',label='Iris-versicolor')

plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='Iris-verginica')

#plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')

plt.legend()

plt.show()