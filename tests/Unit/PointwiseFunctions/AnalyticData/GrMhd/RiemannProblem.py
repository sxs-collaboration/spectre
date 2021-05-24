# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def rest_mass_density(x, adiabatic_index, left_density, right_density,
                      left_pressure, right_pressure, left_velocity,
                      right_velocity, left_magnetic_field,
                      right_magnetic_field, lapse, shift):
    assert len(x) == 3
    return left_density if x[0] <= 0.0 else right_density


def spatial_velocity(x, adiabatic_index, left_density, right_density,
                     left_pressure, right_pressure, left_velocity,
                     right_velocity, left_magnetic_field, right_magnetic_field,
                     lapse, shift):
    return np.asarray(left_velocity if x[0] <= 0.0 else right_velocity)


def specific_internal_energy(x, adiabatic_index, left_density, right_density,
                             left_pressure, right_pressure, left_velocity,
                             right_velocity, left_magnetic_field,
                             right_magnetic_field, lapse, shift):
    return (1.0 / (adiabatic_index - 1.0) * compute_pressure(
        x, adiabatic_index, left_density, right_density, left_pressure,
        right_pressure, left_velocity, right_velocity, left_magnetic_field,
        right_magnetic_field, lapse, shift) / rest_mass_density(
            x, adiabatic_index, left_density, right_density, left_pressure,
            right_pressure, left_velocity, right_velocity, left_magnetic_field,
            right_magnetic_field, lapse, shift))


def compute_pressure(x, adiabatic_index, left_density, right_density,
                     left_pressure, right_pressure, left_velocity,
                     right_velocity, left_magnetic_field, right_magnetic_field,
                     lapse, shift):
    return left_pressure if x[0] <= 0.0 else right_pressure


def lorentz_factor(x, adiabatic_index, left_density, right_density,
                   left_pressure, right_pressure, left_velocity,
                   right_velocity, left_magnetic_field, right_magnetic_field,
                   lapse, shift):
    v = spatial_velocity(x, adiabatic_index, left_density, right_density,
                         left_pressure, right_pressure, left_velocity,
                         right_velocity, left_magnetic_field,
                         right_magnetic_field, lapse, shift)
    return 1. / np.sqrt(1. - np.dot(v, v))


def specific_enthalpy(x, adiabatic_index, left_density, right_density,
                      left_pressure, right_pressure, left_velocity,
                      right_velocity, left_magnetic_field,
                      right_magnetic_field, lapse, shift):
    return (1.0 + adiabatic_index * specific_internal_energy(
        x, adiabatic_index, left_density, right_density, left_pressure,
        right_pressure, left_velocity, right_velocity, left_magnetic_field,
        right_magnetic_field, lapse, shift))


def magnetic_field(x, adiabatic_index, left_density, right_density,
                   left_pressure, right_pressure, left_velocity,
                   right_velocity, left_magnetic_field, right_magnetic_field,
                   lapse, shift):
    return np.asarray(
        left_magnetic_field if x[0] <= 0.0 else right_magnetic_field)


def divergence_cleaning_field(x, adiabatic_index, left_density, right_density,
                              left_pressure, right_pressure, left_velocity,
                              right_velocity, left_magnetic_field,
                              right_magnetic_field, lapse, shift):
    return 0.0
