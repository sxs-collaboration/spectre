# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def compute_rest_mass_density(x, pressure, rest_mass_density, adiabatic_index,
                              advection_velocity, magnetic_field_strength,
                              inner_radius, outer_radius):
    return rest_mass_density


def compute_pressure(x, pressure, rest_mass_density, adiabatic_index,
                     advection_velocity, magnetic_field_strength, inner_radius,
                     outer_radius):
    return pressure


def specific_internal_energy(x, pressure, rest_mass_density, adiabatic_index,
                             advection_velocity, magnetic_field_strength,
                             inner_radius, outer_radius):
    return (1.0 / (adiabatic_index - 1.0) * compute_pressure(
        x, pressure, rest_mass_density, adiabatic_index, advection_velocity,
        magnetic_field_strength, inner_radius, outer_radius) /
            compute_rest_mass_density(x, pressure, rest_mass_density,
                                      adiabatic_index, advection_velocity,
                                      magnetic_field_strength, inner_radius,
                                      outer_radius))


def spatial_velocity(x, pressure, rest_mass_density, adiabatic_index,
                     advection_velocity, magnetic_field_strength, inner_radius,
                     outer_radius):
    return np.array(advection_velocity)


def lorentz_factor(x, pressure, rest_mass_density, adiabatic_index,
                   advection_velocity, magnetic_field_strength, inner_radius,
                   outer_radius):
    v = spatial_velocity(x, pressure, rest_mass_density, adiabatic_index,
                         advection_velocity, magnetic_field_strength,
                         inner_radius, outer_radius)
    return 1. / np.sqrt(1. - np.dot(v, v))


def specific_enthalpy(x, pressure, rest_mass_density, adiabatic_index,
                      advection_velocity, magnetic_field_strength,
                      inner_radius, outer_radius):
    return (1. + specific_internal_energy(
        x, pressure, rest_mass_density, adiabatic_index, advection_velocity,
        magnetic_field_strength, inner_radius, outer_radius) +
            compute_pressure(x, pressure, rest_mass_density, adiabatic_index,
                             advection_velocity, magnetic_field_strength,
                             inner_radius, outer_radius) /
            compute_rest_mass_density(x, pressure, rest_mass_density,
                                      adiabatic_index, advection_velocity,
                                      magnetic_field_strength, inner_radius,
                                      outer_radius))


def magnetic_field(x, pressure, rest_mass_density, adiabatic_index,
                   advection_velocity, magnetic_field_strength, inner_radius,
                   outer_radius):
    radius = np.sqrt(np.square(x[0]) + np.square(x[1]))
    if (radius > outer_radius):
        return np.array([0.0, 0.0, 0.0])
    if (radius < inner_radius):
        return np.array([
            -magnetic_field_strength * x[1] / inner_radius,
            magnetic_field_strength * x[0] / inner_radius, 0.
        ])
    return np.array([
        -magnetic_field_strength * x[1] / radius,
        magnetic_field_strength * x[0] / radius, 0.
    ])


def divergence_cleaning_field(x, pressure, rest_mass_density, adiabatic_index,
                              advection_velocity, magnetic_field_strength,
                              inner_radius, outer_radius):
    return 0.
