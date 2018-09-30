# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions for testing LorentzFactor.cpp


def lorentz_factor(spatial_velocity, spatial_velocity_one_form):
    return 1.0 / np.sqrt(1.0 -
                         np.dot(spatial_velocity, spatial_velocity_one_form))


# End functions for testing LorentzFactor.cpp

# Functions for testing SpecificEnthalpy.cpp


def specific_enthalpy(rest_mass_density, specific_internal_energy, pressure):
    return pressure / rest_mass_density + 1.0 + specific_internal_energy


# End functions for testing SpecificEnthalpy.cpp
