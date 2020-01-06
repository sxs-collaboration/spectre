# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions for testing LorentzFactor.cpp


def lorentz_factor(spatial_velocity, spatial_velocity_one_form):
    return 1.0 / np.sqrt(1.0 -
                         np.dot(spatial_velocity, spatial_velocity_one_form))


# End functions for testing LorentzFactor.cpp

# Functions for testing MassFlux.cpp


def mass_flux(rest_mass_density, spatial_velocity, lorentz_factor, lapse,
              shift, sqrt_det_spatial_metric):
    return rest_mass_density * lorentz_factor * sqrt_det_spatial_metric * \
        (lapse * spatial_velocity - shift)


# End functions for testing MassFlux.cpp

# Functions for testing SpecificEnthalpy.cpp


def relativistic_specific_enthalpy(rest_mass_density, specific_internal_energy,
                                   pressure):
    return pressure / rest_mass_density + 1.0 + specific_internal_energy


# End functions for testing SpecificEnthalpy.cpp
