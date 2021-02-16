# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density(x, mass_density_l, velocity_l, pressure_l, jump_position,
                 epsilon, wave_length, velocity_r, pressure_r,
                 adiabatic_index):
    if x < jump_position:
        return mass_density_l
    return 1.0 + epsilon * np.sin(wave_length * x[0])


def velocity(x, mass_density_l, velocity_l, pressure_l, jump_position, epsilon,
             wave_length, velocity_r, pressure_r, adiabatic_index):
    return np.asarray([velocity_l if x < jump_position else velocity_r])


def pressure(x, mass_density_l, velocity_l, pressure_l, jump_position, epsilon,
             wave_length, velocity_r, pressure_r, adiabatic_index):
    return pressure_l if x < jump_position else pressure_r


def specific_internal_energy(x, mass_density_l, velocity_l, pressure_l,
                             jump_position, epsilon, wave_length, velocity_r,
                             pressure_r, adiabatic_index):
    return pressure(
        x, mass_density_l, velocity_l, pressure_l, jump_position, epsilon,
        wave_length, velocity_r, pressure_r, adiabatic_index) / mass_density(
            x, mass_density_l, velocity_l, pressure_l, jump_position, epsilon,
            wave_length, velocity_r, pressure_r,
            adiabatic_index) / (adiabatic_index - 1.0)
