# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def initial_profile(x):
    result = np.where(x > -0.1, -1.5 * x + 0.85, 1.0)
    result = np.where(x > 0.1, 0.7, result)
    return result


def TildeE(x, t):
    electric_field = x * 0.0
    electric_field[2] = -initial_profile(x[0] - t)
    return electric_field


def TildeB(x, t):
    magnetic_field = x * 0.0
    magnetic_field[0] = 1.0
    magnetic_field[1] = initial_profile(x[0] - t)
    return magnetic_field


def TildePsi(x, t):
    return x[0] * 0


def TildePhi(x, t):
    return x[0] * 0


def TildeQ(x, t):
    return x[0] * 0
