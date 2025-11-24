# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from ConservativeFromPrimitive import b_dot_v, b_squared


def comoving_b_magnitude(
    magnetic_field, spatial_velocity, lorentz_factor, spatial_metric
):
    return np.sqrt(
        b_squared(magnetic_field, spatial_metric) / lorentz_factor**2
        + b_dot_v(magnetic_field, spatial_velocity, spatial_metric) ** 2
    )
