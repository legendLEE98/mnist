import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0, dtype = int)

x = np.arange(-5.0, 5.0, 0.1) # -5 부터 5까지 0.1 단위로
y1 = sigmoid(x)
y2 = step_function(x)
plt.plot(x, y1, label = "sigmoid")
plt.plot(x, y2, linestyle = "--", label = "stair")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()