import numpy as np
from sigmoidGradient import sigmoidGradient
def sigmoid(theta,x):
    return 1.0/(1.0 + np.exp(-np.dot(x,theta.T)))

def nnCostFunction(theta1,theta2,theta3,x,y,lamda = 0):
    m = len(y)
    # Compue Layer 2
    one = np.ones(len(x))
    one = np.reshape(one,(len(x),1))
    a1 = np.concatenate((one,x),axis = 1)
    a2 = sigmoid(theta1,a1)
    # Compute layer 3
    one2 = np.ones(len(a2))
    one2 = np.reshape(one2,(len(a2),1))
    a2 = np.concatenate((one2,a2),axis = 1)
    a3 = sigmoid(theta2,a2)
    # Compute Layer4
    one3 = np.ones(len(a3))
    one3 = np.reshape(one3,(len(a3),1))
    a3 = np.concatenate((one3,a3),axis = 1)
    a4 = sigmoid(theta3,a3)
    # Compute Costfunction with lamda
    j1 = y*np.log(a4)
    j2 = (1-y)*np.log(1-a4)
    j = 1.0/len(y) * (-j1 - j2)
    r1 = np.sum(np.sum(theta1[:,1:]**2)) + np.sum(np.sum(theta2[:,1:]**2)) + np.sum(np.sum(theta3[:,1:]**2))
    J = j.sum() + lamda/(2*len(y))*r1
    # Backpropagation
    delta_4 = a4 - y
    delta_3 = np.dot(delta_4,theta3) * a3 * (1 - a3)
    delta_2 = np.dot(delta_3[:,1:],theta2) * a2 * (1 - a2)
    Delta_1 = 1.0/m * np.dot(delta_2[:,1:].T, a1)
    Delta_2 = 1.0/m * np.dot(delta_3[:,1:].T, a2)
    Delta_3 = 1.0/m * np.dot(delta_4.T, a3)
    return J,Delta_1,Delta_2,Delta_3,a3