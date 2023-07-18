# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def normal_dot_flux(normal, flux_tensor):
    return np.einsum("i,i...", normal, flux_tensor)


def normal_times_flux(normal, flux_tensor):
    return np.tensordot(normal, flux_tensor, axes=0)
