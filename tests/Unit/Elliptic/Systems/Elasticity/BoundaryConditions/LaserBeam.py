# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import sqrt, exp, pi


def normal_dot_minus_stress(x, n, beam_width):
    n /= np.linalg.norm(n)
    r = sqrt(np.linalg.norm(x)**2 - np.dot(x, n)**2)
    beam_profile = exp(-(r / beam_width)**2) / pi / beam_width**2
    return np.tensordot(-n, beam_profile, axes=0)


def normal_dot_minus_stress_linearized(x, n, beam_width):
    return np.zeros(3)
