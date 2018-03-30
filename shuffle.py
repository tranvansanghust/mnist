import numpy as np
def shuffle(x,y):
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    return x[s],y[s]