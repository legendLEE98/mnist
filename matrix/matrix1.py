import numpy as np
import matplotlib.pyplot as plt


a = np.array([[1,2],[3,4]])
print(a.shape)
b = np.array([[5,6],[7,8]])
print(b.shape)


c = np.dot(a,b)

print(c)