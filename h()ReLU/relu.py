import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

a = np.array([1,2,3,4])
print(a)