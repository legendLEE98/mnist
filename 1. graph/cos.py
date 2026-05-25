import numpy as np
import matplotlib.pyplot as plt

# x는 6까지 출력, 0.1 단위 마다
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# sin은 그냥 그리기
plt.plot(x, y1, label = "sin")
# cos 점선으로 그리기
plt.plot(x, y2, linestyle="--", label="cos")

plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()