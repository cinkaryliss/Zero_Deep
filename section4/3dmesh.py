import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def function(x, y):
    return x**2 + y**2

x = np.arange(-3.0, 3.0, 0.1)
y = np.arange(-3.0, 3.0, 0.1)
X, Y = np.meshgrid(x, y)
Z = function(X, Y)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, Z)
plt.show()
