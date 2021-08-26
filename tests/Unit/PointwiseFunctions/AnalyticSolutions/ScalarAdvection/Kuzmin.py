# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

r0 = 0.15


def r_xy(x, x0, y, y0):
    return np.sqrt((x - x0)**2.0 + (y - y0)**2.0) / r0


def u_initial(x_grid):
    x, y = x_grid

    r_cylinder = r_xy(x, 0.5, y, 0.75)
    r_cone = r_xy(x, 0.5, y, 0.25)
    r_hump = r_xy(x, 0.25, y, 0.5)

    if r_cylinder <= 1.0:
        if ((np.abs(x - 0.5) >= 0.025) or (y >= 0.85)):
            u = 1.0
        else:
            u = 0.0
    elif r_cone <= 1.0:
        u = 1.0 - r_cone
    elif r_hump <= 1.0:
        u = 0.25 * (1.0 + np.cos(np.pi * r_hump))
    else:
        u = 0.0

    return u


def u(x, t):
    x0 = (x[0] - 0.5) * np.cos(t) + (x[1] - 0.5) * np.sin(t) + 0.5
    y0 = -(x[0] - 0.5) * np.sin(t) + (x[1] - 0.5) * np.cos(t) + 0.5
    return u_initial([x0, y0])
