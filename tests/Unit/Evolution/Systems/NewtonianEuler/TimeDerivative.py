# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density_cons_flux_impl(
    momentum_density, energy_density, velocity, pressure
):
    return momentum_density


def momentum_density_flux_impl(
    momentum_density, energy_density, velocity, pressure
):
    result = np.outer(momentum_density, velocity)
    result += pressure * np.identity(velocity.size)
    return result


def energy_density_flux_impl(
    momentum_density, energy_density, velocity, pressure
):
    return (energy_density + pressure) * velocity


def mass_density_cons_flux(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    return mass_density_cons_flux_impl(
        momentum_density, energy_density, velocity, pressure
    )


def momentum_density_flux(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    return momentum_density_flux_impl(
        momentum_density, energy_density, velocity, pressure
    )


def energy_density_flux(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    return energy_density_flux_impl(
        momentum_density, energy_density, velocity, pressure
    )


def source_mass_density_cons_uniform_acceleration(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    return mass_density * 0.0


def source_momentum_density_cons_uniform_acceleration(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    if len(momentum_density) == 1:
        return mass_density * np.asarray([0.3])
    elif len(momentum_density) == 2:
        return mass_density * np.asarray([0.3, 1.3])
    elif len(momentum_density) == 3:
        return mass_density * np.asarray([0.3, 1.3, 2.3])


def source_energy_density_cons_uniform_acceleration(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    return np.einsum(
        "i,i->",
        (
            np.asarray([0.3])
            if len(momentum_density) == 1
            else (
                np.asarray([0.3, 1.3])
                if len(momentum_density) == 2
                else np.asarray([0.3, 1.3, 2.3])
            )
        ),
        momentum_density,
    )


_perturbation_amplitude = 0.1


def source_mass_density_cons_vortex_perturbation(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    return mass_density * _perturbation_amplitude * np.cos(coords[2])


def source_momentum_density_cons_vortex_perturbation(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    return momentum_density * _perturbation_amplitude * np.cos(coords[2])


def source_energy_density_cons_vortex_perturbation(
    mass_density, momentum_density, energy_density, velocity, pressure, coords
):
    return (
        (energy_density + pressure + velocity[2] * momentum_density[2])
        * _perturbation_amplitude
        * np.cos(coords[2])
    )
