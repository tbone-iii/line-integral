
# OUTLINE:
# TODO:
# ? Generalize functions for input
# ! 1) Graph function (input function)
# ! 2) 3D function input?
# ! 3)

from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from random import random
from mpl_toolkits.mplot3d import Axes3D


def func(x, y):
    return sqrt(x**2 + y**2)


# Define constants for plotting region
XL = -2
XU = 2
YL = -2
YU = 2
NUM_POINTS = 10

# Determine the gridspace for the plot
x_range = np.linspace(XL, XU, num=NUM_POINTS)
y_range = np.linspace(YL, YU, num=NUM_POINTS)
(xgrid, ygrid) = np.meshgrid(x_range, y_range)

# Determine the points for z given the function
zgrid = xgrid

# Plot the function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=xgrid, ys=ygrid, zs=zgrid)

plt.show()
