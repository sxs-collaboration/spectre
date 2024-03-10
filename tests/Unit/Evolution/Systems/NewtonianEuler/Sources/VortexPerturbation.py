# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def source_mass_density_cons(
    mass_density,
    momentum_density,
    energy_density,
    pressure,
    x,
    perturbation_amplitude,
):
    return perturbation_amplitude * mass_density * np.cos(x[2])


def source_momentum_density(
    mass_density,
    momentum_density,
    energy_density,
    pressure,
    x,
    perturbation_amplitude,
):
    return momentum_density * perturbation_amplitude * np.cos(x[2])


def source_energy_density(
    mass_density,
    momentum_density,
    energy_density,
    pressure,
    x,
    perturbation_amplitude,
):
    return (
        (energy_density + pressure + momentum_density[2] ** 2 / mass_density)
        * perturbation_amplitude
        * np.cos(x[2])
    )
