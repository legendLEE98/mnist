import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype = int)

