import numpy as np
def unRolling(theta):
    m = len(theta[0])
    n = len(theta)
    return np.reshape(theta,(m*n,1))