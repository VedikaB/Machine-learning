import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis=0
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    hypothesis=theta[0]*X[i][0]+theta[1]*X[i][1]+theta[2]*(pow(X[i][1],2))+theta[3]*(pow(X[i][1],3))+theta[4]*(pow(X[i][1],4))+theta[5]*(pow(X[i][1],5))
    ########################################/
    
    return hypothesis
