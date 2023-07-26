# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def scalar_charge_integrand(phi, unit_normal_vector):
    result = np.dot(phi, unit_normal_vector)
    result /= -4.0 * np.pi
    return result
