import numpy as np
import matplotlib.pyplot as plt

x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x > 0
print(y)

y = y.astype(int)
print(y)