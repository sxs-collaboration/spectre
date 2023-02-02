# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def initial_profile(x):
    result = np.where(x > -0.1, -1.5 * x + 0.85, 1.0)
    result = np.where(x > 0.1, 0.7, result)
    return result


def TildeE(x):
    electric_field = x * 0.0
    electric_field[1] = 0.5
    electric_field[2] = -0.5
    return electric_field


def TildeB(x):
    magnetic_field = x * 0.0
    magnetic_field[0] = 1.0
    magnetic_field[1] = np.where(x[0] < -0.1, 1.0, -10 * x[0])
    magnetic_field[1] = np.where(x[0] > 0.1, -1.0, magnetic_field[1])
    magnetic_field[2] = magnetic_field[1]
    return magnetic_field


def TildePsi(x):
    return x[0] * 0


def TildePhi(x):
    return x[0] * 0


def TildeQ(x):
    return x[0] * 0
