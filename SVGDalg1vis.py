import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import scipy.stats
from matplotlib import cm

n = 111
x = torch.empty((3, n * n))
for i in range(n):
    for j in range(n):
        x[0, i * n + j] = (i - np.floor(n / 2)) / n * 2 * 5
        x[1, i * n + j] = (j - np.floor(n / 2)) / n * 2 * 5
        x[2, i * n + j] = 0


def f(x, y):
    return 1.0 / (2 * np.pi) * np.exp(-1.0 / 2.0 * (np.power(x, 2) + np.power(y, 2)))


x1 = np.linspace(-5, 5, 100)
y1 = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x1, y1)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor="green", linewidth=0, alpha=0.7)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.scatter(x[0, :], x[1, :], x[2, :], color="#30FF30")
plt.show()
