# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def compute_pi(deriv, shift, lapse):
    return np.einsum("i,i", shift, deriv) / lapse
