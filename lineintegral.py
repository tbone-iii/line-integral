
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


def func2(x: Array, y: Array) -> Array:
    """ 3D function f(x, y) """
    # sinc curve
    # https://qph.fs.quoracdn.net/main-qimg-8ba30cbe0b2ee5b98e898e4d38cf37b7-c

    f = np.sqrt(x**2+y**2) * np.sinc(np.sqrt(x**2+y**2)) + 0.3

    return f


def main():
    # Define constants for plotting region
    (XL, XU) = (-3, 3)
    (YL, YU) = (-3, 3)
    (TL, TU) = (-3**(1/3), 3**(1/3))
    TSIZE = 10**3
    NUM_POINTS = 1000

    # Determine the gridspace for the plot
    x_range = np.linspace(XL, XU, num=NUM_POINTS)
    y_range = np.linspace(YL, YU, num=NUM_POINTS)

    # TODO: Fix the logspace x_range and y_range. Crashing for unknown reason.
    # temp = np.log(np.arange(1, 100))[1:]
    # x_range = -temp/temp[-1]
    # x_range = np.concatenate((x_range,
    #                           np.linspace(-0.12, 0.12, NUM_POINTS),
    #                           -x_range))
    # y_range = x_range

    t_range = np.linspace(TL, TU, num=TSIZE)
    (xgrid, ygrid) = np.meshgrid(x_range, y_range)

    area = our_integrator(func=func2, curve=curve, domain=t_range)
    print(f"\nArea: {bcolors.OKGREEN}{area: .8f}{bcolors.ENDC}\n")

    # Determine the points for z given the function
    zgrid = func2(xgrid, ygrid)

    # Plot the function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid, ygrid, zgrid, cmap=cm.coolwarm,
                    linewidth=0)

    # Plot the projection
    (xt, yt) = curve(t_range)
    # x_indices = np.where(abs(xt) < 1)
    # y_indices = np.where(abs(yt) < 1)
    # indices = np.intersect1d(x_indices, y_indices)

    # xt, yt = xt[indices], yt[indices]
    ax.plot(xt, yt, 'b--')

    # Plot the curve on the function
    zt = func2(xt, yt)
    ax.plot(xt, yt, zt, 'k')

    # Adjust the plot
    ax.set_aspect("equal")
    plt.xlim((XL*1.1, XU*1.1))
    plt.ylim((YL*1.1, YU*1.1))
    ax.set_zlim(0, 1.5)

    # Indicate xyz axes
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.set_zlabel(r"z")
    ax.set_title(r"Hemisphere $\oint_C f(s) \,ds$"+f" = {area: 0.4f}")

    # LAST STATEMENT
    plt.show()


if __name__ == '__main__':
    main()
