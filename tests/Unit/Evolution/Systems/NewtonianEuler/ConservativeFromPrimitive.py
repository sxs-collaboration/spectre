# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density_cons(mass_density, velocity, specific_internal_energy):
    return mass_density


def momentum_density(mass_density, velocity, specific_internal_energy):
    return mass_density * velocity


def energy_density(mass_density, velocity, specific_internal_energy):
    return (
        0.5 * mass_density * np.dot(velocity, velocity)
        + mass_density * specific_internal_energy
    )
