# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions for testing EnergyDensity.cpp


def energy_density(pi, phi):
    return 0.5 * (pi * pi + np.linalg.norm(phi) * np.linalg.norm(phi))


# End functions for testing EnergyDensity.cpp
