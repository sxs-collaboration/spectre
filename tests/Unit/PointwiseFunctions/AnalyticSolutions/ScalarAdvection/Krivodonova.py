# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def func_F(x, alpha, a):
    return np.sqrt(max(1.0 - alpha**2.0 * (x - a)**2.0, 0.0))


def func_G(x, beta, z):
    return np.exp(-beta * (x - z)**2.0)


def u_initial(x_grid):
    a = 0.5
    z = -0.7
    delta = 0.005
    alpha = 10.0
    beta = np.log(2.0) / (36.0 * delta**2)

    x0 = x_grid[0]
    if ((-0.8 <= x0) and (x0 <= -0.6)):
        u = (func_G(x0, beta, z - delta) + func_G(x0, beta, z + delta) +
             4.0 * func_G(x0, beta, z)) / 6.0
    elif ((-0.4 <= x0) and (x0 <= -0.2)):
        u = 1.0
    elif ((0.0 <= x0) and (x0 <= 0.2)):
        u = 1.0 - np.abs(10.0 * x0 - 1.0)
    elif ((0.4 <= x0) and (x0 <= 0.6)):
        u = (func_F(x0, alpha, a - delta) + func_F(x0, alpha, a + delta) +
             4.0 * func_F(x0, alpha, a)) / 6.0
    else:
        u = 0.0

    return u


def u(x, t):
    xt = x - t
    x0 = xt - 2. * np.floor((xt + 1.) * 0.5)
    return u_initial(x0)
