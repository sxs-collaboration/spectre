# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Test function for computing flux
def compute_flux(u, velocity_field):
    return u * velocity_field


# Test function for computing advection velocity field
def velocity_field(coords):
    if len(coords) == 1:
        return np.ones(1)
    elif len(coords) == 2:
        return np.asarray([0.5 - coords[1], -0.5 + coords[0]])
    else:
        raise TypeError("Coordinate dimension does not match")
