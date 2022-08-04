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
    no_of_params= len(X[i])
    k = 0
    while k < no_of_params:
        hypothesis = hypothesis + (theta[k]*X[i][k])
        k+=1
    ########################################/
    
    return hypothesis
