# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def ricci_tensor(christoffel, deriv_christoffel):
    return (np.einsum("ccab", deriv_christoffel) - 0.5 * (np.einsum(
        "bcac", deriv_christoffel) + np.einsum("acbc", deriv_christoffel)) +
        np.einsum("dab,ccd", christoffel, christoffel) -
        np.einsum("dac,cbd", christoffel, christoffel))
