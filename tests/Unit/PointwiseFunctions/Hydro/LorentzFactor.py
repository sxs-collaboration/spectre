# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def lorentz_factor(spatial_velocity, spatial_velocity_one_form):
    return 1.0 / np.sqrt(
        1.0 - np.dot(spatial_velocity.T, spatial_velocity_one_form)
    )
