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
        return np.abs(x[0] - 0.5) * (20. * (x[0] - 0.5)**2 - 1.5)
    elif len(x) == 2:
        r = np.linalg.norm(x - 0.5)
        return r * (
            -0.5625 + 6.25 * r**2 - 6.125 * r**4 + 4.125 * (x[0] - 0.5)**4 \
            - 24.75 * (x[0] - 0.5)**2 * (x[1] - 0.5)**2                    \
            + 4.125 * (x[1] - 0.5)**4)

def auxiliary_source(x):
    return np.zeros(len(x))
# End functions for testing ProductOfSinusoids.cpp
