import numpy as np
def sigmoidGradient(z):
    return 1.0/(1 + np.exp(-z)) * (1 - 1.0/(1 + np.exp(-z)))