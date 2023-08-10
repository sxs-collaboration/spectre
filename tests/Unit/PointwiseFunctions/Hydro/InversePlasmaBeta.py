# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions testing InversePlasmaBeta


def inverse_plasma_beta(comoving_magnetic_field_magnitude, fluid_pressure):
    return 0.5 * (comoving_magnetic_field_magnitude**2) / fluid_pressure


# End function for testing InversePlasmaBeta
