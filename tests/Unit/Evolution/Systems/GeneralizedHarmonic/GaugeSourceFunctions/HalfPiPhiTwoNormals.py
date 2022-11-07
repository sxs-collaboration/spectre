# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def half_pi_two_normals(normal_vector, pi, phi):
    return 0.5 * np.einsum("a,b,ab->", normal_vector, normal_vector, pi)


def half_phi_two_normals(normal_vector, pi, phi):
    return 0.5 * np.einsum("a,b,iab->i", normal_vector, normal_vector, phi)
