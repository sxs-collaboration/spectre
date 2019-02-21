# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def compute_piecewise(x, inner_radius, outer_radius, inner_value, outer_value):
    radius = np.sqrt(np.square(x[0]) + np.square(x[1]))
    if (radius > outer_radius):
        return outer_value
    elif (radius < inner_radius):
        return inner_value
    else:
        piecewise_scalar = (-1.0 * radius + inner_radius) * \
            np.log(outer_value)
        piecewise_scalar += (radius - outer_radius) * np.log(inner_value)
        piecewise_scalar /= inner_radius - outer_radius
        return np.exp(piecewise_scalar)


def rest_mass_density(x, inner_radius, outer_radius, inner_density,
                      outer_density, inner_pressure, outer_pressure,
                      magnetic_field, adiabatic_index):
    return compute_piecewise(x, inner_radius, outer_radius, inner_density,
                             outer_density)


def spatial_velocity(x, inner_radius, outer_radius, inner_density,
                     outer_density, inner_pressure, outer_pressure,
                     magnetic_field, adiabatic_index):
    return np.zeros(3)


def specific_internal_energy(x, inner_radius, outer_radius, inner_density,
                             outer_density, inner_pressure, outer_pressure,
                             magnetic_field, adiabatic_index):
    return (1.0 / (adiabatic_index - 1.0) *
            pressure(x, inner_radius, outer_radius, inner_density,
                     outer_density, inner_pressure, outer_pressure,
                     magnetic_field, adiabatic_index) /
            rest_mass_density(x, inner_radius, outer_radius, inner_density,
                              outer_density, inner_pressure, outer_pressure,
                              magnetic_field, adiabatic_index))


def pressure(x, inner_radius, outer_radius, inner_density, outer_density,
             inner_pressure, outer_pressure, magnetic_field, adiabatic_index):
    return compute_piecewise(x, inner_radius, outer_radius, inner_pressure,
                             outer_pressure)


def lorentz_factor(x, inner_radius, outer_radius, inner_density, outer_density,
                   inner_pressure, outer_pressure, magnetic_field,
                   adiabatic_index):
    return 1.0


def specific_enthalpy(x, inner_radius, outer_radius, inner_density,
                      outer_density, inner_pressure, outer_pressure,
                      magnetic_field, adiabatic_index):
    return (1.0 + adiabatic_index *
            specific_internal_energy(x, inner_radius, outer_radius,
                                     inner_density, outer_density,
                                     inner_pressure, outer_pressure,
                                     magnetic_field, adiabatic_index))


def magnetic_field(x, inner_radius, outer_radius, inner_density, outer_density,
                   inner_pressure, outer_pressure, magnetic_field,
                   adiabatic_index):
    return np.array(magnetic_field)


def divergence_cleaning_field(x, inner_radius, outer_radius, inner_density,
                              outer_density, inner_pressure, outer_pressure,
                              magnetic_field, adiabatic_index):
    return 0.0
