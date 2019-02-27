import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot for easy21
    """
    x_range = np.arange(1, 22)
    y_range = np.arange(1, 10)
    Y, X = np.meshgrid(y_range, x_range)

    # Find value for all (x, y) coordinates
    Z = V[10:31, 0:9]

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z, title)
