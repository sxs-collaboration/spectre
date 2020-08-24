# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def spherical_metric(sin_theta, cos_theta, t, r, m, f):
    return np.array([[(-(1.0 - 2.0 * m / r - f**2 * r**2 * sin_theta**2)), 0.0,
                      0.0, r**2 * f * sin_theta],
                     [0.0, 1.0 / (1.0 - 2.0 * m / r), 0.0, 0.0],
                     [0.0, 0.0, r**2, 0.0],
                     [r**2 * f * sin_theta, 0.0, 0.0, r**2]])


def dr_spherical_metric(sin_theta, cos_theta, t, r, m, f):
    return np.array([[(-(2.0 * m / r**2 - 2.0 * f**2 * r * sin_theta**2)), 0.0,
                      0.0, 2.0 * r * f * sin_theta],
                     [0.0, -2.0 * m / (r - 2.0 * m)**2, 0.0, 0.0],
                     [0.0, 0.0, 2.0 * r, 0.0],
                     [2.0 * r * f * sin_theta, 0.0, 0.0, 2.0 * r]])


def dt_spherical_metric(sin_theta, cos_theta, t, r, m, f):
    return np.zeros((4, 4))


def news(sin_theta, t, r, m, f):
    return complex(0.0, 0.0)
