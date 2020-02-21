# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def compute_piecewise(x, rotor_radius, inner_value, outer_value):
    radius = np.sqrt(np.square(x[0]) + np.square(x[1]))
    if (radius > rotor_radius):
        return outer_value
    else:
        return inner_value


def rest_mass_density(x, rotor_radius, inner_density, outer_density, pressure,
                      angular_velocity, magnetic_field, adiabatic_index):
    return compute_piecewise(x, rotor_radius, inner_density, outer_density)


def spatial_velocity(x, rotor_radius, inner_density, outer_density, pressure,
                     angular_velocity, magnetic_field, adiabatic_index):
    omega = compute_piecewise(x, rotor_radius, angular_velocity, 0.0)
    return np.array([-x[1] * omega, x[0] * omega, 0.0])


def specific_internal_energy(x, rotor_radius, inner_density, outer_density,
                             pressure, angular_velocity, magnetic_field,
                             adiabatic_index):
    return (1.0 / (adiabatic_index - 1.0) * compute_pressure(
        x, rotor_radius, inner_density, outer_density, pressure,
        angular_velocity, magnetic_field, adiabatic_index) / rest_mass_density(
            x, rotor_radius, inner_density, outer_density, pressure,
            angular_velocity, magnetic_field, adiabatic_index))


def compute_pressure(x, rotor_radius, inner_density, outer_density, pressure,
                     angular_velocity, magnetic_field, adiabatic_index):
    return pressure


def lorentz_factor(x, rotor_radius, inner_density, outer_density, pressure,
                   angular_velocity, magnetic_field, adiabatic_index):
    v = spatial_velocity(x, rotor_radius, inner_density, outer_density,
                         pressure, angular_velocity, magnetic_field,
                         adiabatic_index)
    return 1. / np.sqrt(1. - np.dot(v, v))


def specific_enthalpy(x, rotor_radius, inner_density, outer_density, pressure,
                      angular_velocity, magnetic_field, adiabatic_index):
    return (1.0 + adiabatic_index * specific_internal_energy(
        x, rotor_radius, inner_density, outer_density, pressure,
        angular_velocity, magnetic_field, adiabatic_index))


def magnetic_field(x, rotor_radius, inner_density, outer_density, pressure,
                   angular_velocity, magnetic_field, adiabatic_index):
    return np.array(magnetic_field)


def divergence_cleaning_field(x, rotor_radius, inner_density, outer_density,
                              pressure, angular_velocity, magnetic_field,
                              adiabatic_index):
    return 0.0
