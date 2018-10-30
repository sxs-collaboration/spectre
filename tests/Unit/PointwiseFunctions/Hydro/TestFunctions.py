# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing ComovingMagneticField.cpp
def comoving_magnetic_field(eulerian_b_field, transport_velocity,
                            spatial_velocity_oneform, lorentz_factor, lapse):
    result = np.zeros(transport_velocity.size + 1)
    result[0] = (lorentz_factor *
                 np.dot(eulerian_b_field, spatial_velocity_oneform)) / lapse
    result[1:] = (eulerian_b_field[0:] / lorentz_factor +
                  result[0] * transport_velocity[0:])
    return result


def comoving_magnetic_field_squared(eulerian_b_field, eulerian_b_field_squared,
                                    spatial_velocity_oneform, lorentz_factor):
    return (eulerian_b_field_squared / lorentz_factor**2 +
            np.dot(eulerian_b_field, spatial_velocity_oneform)**2)


# End functions for testing ComovingMagneticField.cpp

# Functions for testing LorentzFactor.cpp


def lorentz_factor(spatial_velocity, spatial_velocity_one_form):
    return 1.0 / np.sqrt(1.0 -
                         np.dot(spatial_velocity, spatial_velocity_one_form))


# End functions for testing LorentzFactor.cpp

# Functions for testing SpecificEnthalpy.cpp


def specific_enthalpy(rest_mass_density, specific_internal_energy, pressure):
    return pressure / rest_mass_density + 1.0 + specific_internal_energy


# End functions for testing SpecificEnthalpy.cpp

# Functions for testing TransportVelocity.cpp


def transport_velocity(spatial_velocity, lapse, shift):
    return lapse * spatial_velocity[0:] - shift[0:]


# End functions for testing TransportVelocity.cpp
