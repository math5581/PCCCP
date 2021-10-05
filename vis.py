import numpy as np
import matplotlib.pyplot as plt


x1 = np.array([1, 2, 3, 4])
x2 = np.array([1, 2, 3, 4]) + 3
x3 = np.array([3, 5, 8, 12])



y1 = np.array([1, 1.25, 1.4, 1.45])
y2 = np.array([1, 1.25, 1.4, 1.45]) * 20
y3 = np.array([1, 1.25, 1.4, 1.45]) * 120


plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)

plt.grid()
plt.show()
