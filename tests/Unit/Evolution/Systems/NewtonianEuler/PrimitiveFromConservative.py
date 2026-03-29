# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density(mass_density_cons, momentum_density, energy_density):
    return mass_density_cons


def velocity(mass_density_cons, momentum_density, energy_density):
    return momentum_density / mass_density_cons


def specific_internal_energy(
    mass_density_cons, momentum_density, energy_density
):
    veloc = velocity(mass_density_cons, momentum_density, energy_density)
    return energy_density / mass_density_cons - 0.5 * np.dot(veloc, veloc)


def pressure_1d(mass_density_cons, momentum_density, energy_density):
    return 1.4 * np.power(mass_density_cons, 5.0 / 3.0)


def pressure_2d(mass_density_cons, momentum_density, energy_density):
    return (
        (2.0 / 3.0)
        * mass_density_cons
        * specific_internal_energy(
            mass_density_cons, momentum_density, energy_density
        )
    )
