# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from scipy.special import sph_harm


def pi(x, radius, width, l, m):
    radial = np.exp(-(np.linalg.norm(x) - radius)**2 / width**2)
    # note opposite theta, phi convention for Spectre and scipy
    phi = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])
    theta = np.arctan2(x[1], x[0])
    angular = sph_harm(m, l, theta, phi)
    return radial * np.real(angular)
