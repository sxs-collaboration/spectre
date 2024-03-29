# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def perturbation_profile(z):
    return np.sin(z)


def deriv_of_perturbation_profile(z):
    return np.cos(z)


def mass_density(
    x,
    t,
    adiabatic_index,
    center,
    mean_velocity,
    strength,
    perturbation_amplitude,
):
    x_tilde = x - center - t * np.array(mean_velocity)
    temp = 1.0 - strength * strength * (adiabatic_index - 1.0) * np.exp(
        1.0 - np.dot(x_tilde[:2], x_tilde[:2])
    ) / (8.0 * adiabatic_index * np.pi**2)
    return np.power(temp, 1.0 / (adiabatic_index - 1.0))


def velocity(
    x,
    t,
    adiabatic_index,
    center,
    mean_velocity,
    strength,
    perturbation_amplitude,
):
    x_tilde = x - center - t * np.array(mean_velocity)
    temp = (
        0.5
        * strength
        * np.exp(0.5 * (1.0 - np.dot(x_tilde[:2], x_tilde[:2])))
        / np.pi
    )
    velocity = np.copy(mean_velocity)
    velocity[0] -= x_tilde[1] * temp
    velocity[1] += x_tilde[0] * temp
    if velocity.size == 3:
        velocity[2] += perturbation_amplitude * perturbation_profile(x[2])
    return velocity


def specific_internal_energy(
    x,
    t,
    adiabatic_index,
    center,
    mean_velocity,
    strength,
    perturbation_amplitude,
):
    return np.power(
        mass_density(
            x,
            t,
            adiabatic_index,
            center,
            mean_velocity,
            strength,
            perturbation_amplitude,
        ),
        adiabatic_index - 1.0,
    ) / (adiabatic_index - 1.0)


def pressure(
    x,
    t,
    adiabatic_index,
    center,
    mean_velocity,
    strength,
    perturbation_amplitude,
):
    return np.power(
        mass_density(
            x,
            t,
            adiabatic_index,
            center,
            mean_velocity,
            strength,
            perturbation_amplitude,
        ),
        adiabatic_index,
    )
