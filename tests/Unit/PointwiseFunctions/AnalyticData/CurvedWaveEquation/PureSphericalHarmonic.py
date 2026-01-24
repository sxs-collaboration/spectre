# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

try:
    from scipy.special import sph_harm_y
except ImportError:
    # SciPy < 1.15
    from scipy.special import sph_harm

    def sph_harm_y(n, m, theta, phi):
        return sph_harm(m, n, phi, theta)


def pi(x, radius, width, l, m):
    radial = np.exp(-((np.linalg.norm(x) - radius) ** 2) / width**2)
    theta = np.arctan2(np.sqrt(x[0] ** 2 + x[1] ** 2), x[2])
    phi = np.arctan2(x[1], x[0])
    angular = sph_harm_y(l, m, theta, phi)
    return radial * np.real(angular)
