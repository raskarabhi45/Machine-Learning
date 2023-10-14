#Machine Learning Case studies start
#pip install scikit-learn
#import sklearn
#import required libraries
from sklearn import tree
#With encoding 
#Load the dataset
# Rough   1
# Smooth  0

# Tennis   1
# Cricket  2

Features=[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]

Labels=[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

#Decide the Machine Learning Algorithm
obj=tree.DecisionTreeClassifier()

#Perform the training of model 
obj=obj.fit(Features,Labels) #Fit method internally perform Training... 

#Perform the testing
print(obj.predict([[97,0],[35,1]]))