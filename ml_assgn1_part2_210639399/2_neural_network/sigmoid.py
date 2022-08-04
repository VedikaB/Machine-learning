import numpy as np

def sigmoid(z):
    
    output = 0.0
    #########################################
    # Write your code here
    # modify this to return z passed through the sigmoid function
    output = (np.exp(z) / (np.exp(z) + 1))
    ########################################/
    
    return output
