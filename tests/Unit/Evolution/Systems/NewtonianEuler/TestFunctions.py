# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density_flux(momentum_density, energy_density, velocity, pressure):
    return momentum_density


def momentum_density_flux(momentum_density, energy_density, velocity, pressure):
    result = np.outer(momentum_density, velocity)
    result += pressure * np.identity(velocity.size)
    return result


def energy_density_flux(momentum_density, energy_density, velocity, pressure):
    return (energy_density + pressure) * velocity
