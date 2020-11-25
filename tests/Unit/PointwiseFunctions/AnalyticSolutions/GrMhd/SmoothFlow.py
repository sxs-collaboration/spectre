# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Hydro.SmoothFlow import *


def magnetic_field(x, t, mean_velocity, wave_vector, pressure, adiabatic_index,
                   density_amplitude):
    return np.array([0.0, 0.0, 0.0])


def divergence_cleaning_field(x, t, mean_velocity, wave_vector, pressure,
                              adiabatic_index, density_amplitude):
    return 0.0
