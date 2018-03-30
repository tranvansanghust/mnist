import numpy as np
from nnCostFunction import sigmoid

def predict(theta1, theta2, theta3, x):
    # Compue Layer 2
    one = np.ones(len(x))
    one = np.reshape(one,(len(x),1))
    # print one
    a1 = np.concatenate((one,x),axis = 1)
    a2 = sigmoid(theta1,a1)
    # Compute Layer3
    one2 = np.ones(len(a2))
    one2 = np.reshape(one2,(len(a2),1))
    a2 = np.concatenate((one2,a2),axis = 1)
    a3 = sigmoid(theta2,a2)
    # Compute Layer4
    one3 = np.ones(len(a3))
    one3 = np.reshape(one3,(len(a3),1))
    a3 = np.concatenate((one3,a3),axis = 1)
    a4 = sigmoid(theta3,a3)
    return a4