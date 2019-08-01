
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable
from mpl_toolkits import mplot3d     # NOQA implicit import
from matplotlib import cm

# Define array type
Array = np.ndarray


def compute_difference(t: Array) -> Array:
    """ Calculates the difference in 't' for each 't' in the array.
        Returns an array with 1 less length than the original array."""
    # Shifts over t with a 0 in front, removing the last element
    t_temp = t[:-1]
    t_shifted = np.insert(t_temp, 0, 0)

    return (t - t_shifted)


# Integrator methods
def our_integrator(func: Callable, curve: Callable, domain: Array) -> float:
    """ Integrates along the curve to find area beneath function given the
        domain. Uses trapezoidal rule. Details are further in the description.
        'func':   the 3D function f(x, y)
        'curve':  the 2D curve on the xy-plane (parametric)
        'domain': the array of points parameter 't' to integrate along

        dt -> x(t1+dt)-x(t1) = dx
           -> y(t1+dt)-y(t1) = dy
        INT{f(t)} = 1/2 * sqrt(dx^2+dy^2) * [f(t1) + f(t1+dt)]
    """
    # Determine (x, y) points and find their dx/dy values
    (x, y) = curve(domain)
    dx = compute_difference(x)
    dy = compute_difference(y)

    # Approximate distance between points (used as 'height' in trap. rule)
    height = np.sqrt(dx**2 + dy**2)

    # Area of trapezoid for use in trapezoid rule for integration
    areas = (1/2) * height * func(x, y)
    return sum(areas)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def curve(t: Array) -> Tuple[Array, Array]:
    """ Defined parametric curve C to integrate along (line integral).
        Returns a tuple of two array types given an array input.
    """
    x = np.sin(t)
    y = t**3
    return (x, y)


def func(x: Array, y: Array) -> Array:
    """ 3D function f(x, y) """
    # x^2 + y^2 + z^2 = r^2   ----> z = (r^2 - x^2 - y^2)^(1/2)
    radius = 1
    discriminant = radius**2 - x**2 - y**2
    discriminant[np.where(discriminant < 0)] = 0

    return np.sqrt(discriminant)


def main():
    # Define constants for plotting region
    (XL, XU) = (-1, 1)
    (YL, YU) = (-1, 1)
    (TL, TU) = (-2, 2)
    TSIZE = 50
    NUM_POINTS = 100

    # Determine the gridspace for the plot
    # x_range = np.linspace(XL, XU, num=NUM_POINTS)
    # y_range = np.linspace(YL, YU, num=NUM_POINTS)

    # TODO: Fix the logspace x_range and y_range. Crashing for unknown reason.

    x_range = -np.log(np.arange(1, 100))[1:]/4.5
    x_range = np.concatenate((x_range, np.array([0]), np.log(-x_range)))
    y_range = x_range

    t_range = np.linspace(TL, TU, num=TSIZE)
    (xgrid, ygrid) = np.meshgrid(x_range, y_range)

    area = our_integrator(func=func, curve=curve, domain=t_range)
    print(f"\nArea: {bcolors.OKGREEN}{area: .4f}{bcolors.ENDC}\n")

    # Determine the points for z given the function
    zgrid = func(xgrid, ygrid)

    # Plot the function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid, ygrid, zgrid, cmap=cm.coolwarm,
                    linewidth=0)

    # Plot the projection
    (xt, yt) = curve(t_range)
    x_indices = np.where(abs(xt) < 1)
    y_indices = np.where(abs(yt) < 1)
    indices = np.intersect1d(x_indices, y_indices)

    xt, yt = xt[indices], yt[indices]
    ax.plot(xt, yt, 'b--')

    # Plot the curve on the function
    zt = func(xt, yt)
    ax.plot(xt, yt, zt, 'k')

    # LAST STATEMENT
    plt.show()


if __name__ == '__main__':
    main()
