import numpy as np
def randInitializeWeights(L_in, L_out):
    epsilon = np.sqrt(6)/(np.sqrt(L_in + L_out))
    theta = np.random.rand(L_out,1 + L_in) * (2 * epsilon) - epsilon
    return theta