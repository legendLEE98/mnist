import numpy as np
import matplotlib.pyplot as plt

def NAND(x1, x2):
    x = np.array([x1, x2])
    y = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x*y) + b

    if tmp > 0:
        return 1
    if tmp <= 0:
        return 0


print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))