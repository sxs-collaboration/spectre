# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing IsentropicVortexSource.cpp
def vortex_mass_density_source(mass_density, momentum_density, energy_density,
                               pressure, velocity_z, dz_velocity_z):
    return mass_density * dz_velocity_z


def vortex_momentum_density_source(mass_density, momentum_density,
                                   energy_density, pressure, velocity_z,
                                   dz_velocity_z):
    result = momentum_density * dz_velocity_z
    result[2] *= 2.0
    return result


def vortex_energy_density_source(mass_density, momentum_density, energy_density,
                                 pressure, velocity_z, dz_velocity_z):
    return ((energy_density + pressure + momentum_density[2] * velocity_z) *
            dz_velocity_z)


# End functions for testing IsentropicVortexSource.cpp
