#Linear Regression with user defined Algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def MarvellousHeadBrainPredictor():

    #Load data
    X=[1,2,3,4,5]
    Y=[3,4,2,4,5]

    print("Values of Independent variables x ",X)
    print("Values if dependent variables y ",Y)

    #Least square method
    mean_x=np.mean(X)
    mean_y=np.mean(Y)

    print("Mean of independent variable x ",mean_x)
    print("Mean of dependent variable y ",mean_y)

    n=len(X)

    numerator=0
    denominator=0

    # Equation of line is  y = mx + c

    for i in range(n):
        numerator+=(X[i]-mean_x)*(Y[i]-mean_y)
        denominator+=(X[i]-mean_x)**2

        m=numerator/denominator

        # c = y` - mx`

        c=mean_y-(m*mean_x)

        print("Slope of regression line is ",m)
        print("Y intercept of regression line is ",c)

        max_x=np.max(X)+100
        min_x=np.min(X)-100

        #Dispaly Plotting of above points
        x=np.linspace(min_x,max_x,n)

        y = m*x + c

        plt.plot(x,y,color='#58b970',label='Regression Line')

        plt.scatter(X,Y,color='#ef5423',label='scatter plot')

        plt.xlabel('Head size inn cm3')

        plt.ylabel('Brain weight in gram')

        plt.legend()
        plt.show()

        #FIindout goodness of fit ie R square

        ss_t=0
        ss_r=0

        for i in range(n):
            y_pred=c+m*X[i]
            ss_t+=(Y[i]-mean_y)**2
            ss_r+=(Y[i]-y_pred)**2

        r2=1-(ss_r/ss_t)

        print(r2)




def main():
    print("-----Marvellous Infosytems by Piyush Khairnar------")

    print("Supervised Machine Learning")

    print("Linear regression on Head and Brain size data set")

    MarvellousHeadBrainPredictor()



if __name__=="__main__":
    main()