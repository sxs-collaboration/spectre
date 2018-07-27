# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing ProductOfSinusoids.cpp
def poly(x, coeffs):
    return np.sum(c * x**p for p, c in enumerate(coeffs))

def field(x):
    return np.prod(x * (1. - x)) * np.sum((x - 0.5)**2)**(3. / 2.)

def auxiliary_field(x):
    if len(x) == 1:
        return np.abs(x - 0.5) * poly(x, [0.25, -3, 7.5, -5])
    elif len(x) == 2:
        return np.sqrt(np.sum((x - 0.5)**2)) * np.array([
            x[d - 1] * (1. - x[d - 1]) * (poly(x[d], [
                0.5, poly(x[d - 1], [-3.5, 2, -2]), 7.5, -5
                ]) + poly(x[d - 1], [0, -1, 1]))
            for d in range(len(x))])

def source(x):
    if len(x) == 1:
        return (poly(x, [0.875, -8.5, 28.5, -40., 20.]) / np.abs(x - 0.5))[0]
    elif len(x) == 2:
        return poly(x[0], [
            poly(x[1], [0., 2., -7., 12., -11., 6., -2.]),
            poly(x[1], [2., -26.5, 65.5, -78., 39.]),
            poly(x[1], [-7., 65.5, -104.5, 78., -39.]),
            poly(x[1], [12., -78., 78.]),
            poly(x[1], [-11., 39., -39.]),
            6., -2.
        ]) / np.linalg.norm(x - 0.5)

def auxiliary_source(x):
    return np.zeros(len(x))
# End functions for testing ProductOfSinusoids.cpp
