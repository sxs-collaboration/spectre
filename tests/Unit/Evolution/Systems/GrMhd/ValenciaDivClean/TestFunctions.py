# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def b_dot_v(magnetic_field, spatial_velocity, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(magnetic_field, spatial_velocity))


def b_squared(magnetic_field, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(magnetic_field, magnetic_field))


def magnetic_field_one_form(magnetic_field, spatial_metric):
    return np.einsum("a, ia", magnetic_field, spatial_metric)


def p_star(pressure, b_dot_v, b_squared, lorentz_factor):
    return pressure + 0.5 * (b_dot_v**2 + b_squared / lorentz_factor**2)


def spatial_velocity_one_form(spatial_velocity, spatial_metric):
    return np.einsum("a, ia", spatial_velocity, spatial_metric)


def vsq(spatial_velocity, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(spatial_velocity, spatial_velocity))


# Functions for testing Characteristics.cpp
def characteristic_speeds(lapse, shift, spatial_velocity,
                          spatial_velocity_sqrd, sound_speed_sqrd,
                          alfven_speed_sqrd, normal_oneform):
    normal_velocity = np.dot(spatial_velocity, normal_oneform)
    normal_shift = np.dot(shift, normal_oneform)
    sound_speed_sqrd += alfven_speed_sqrd * (1.0 - sound_speed_sqrd)
    prefactor = lapse / (1.0 - spatial_velocity_sqrd * sound_speed_sqrd)
    first_term = prefactor * normal_velocity * (1.0 - sound_speed_sqrd)
    second_term = (prefactor * np.sqrt(sound_speed_sqrd) * np.sqrt(
        (1.0 - spatial_velocity_sqrd) *
        (1.0 - spatial_velocity_sqrd * sound_speed_sqrd -
         normal_velocity * normal_velocity * (1.0 - sound_speed_sqrd))))
    result = [-lapse - normal_shift]
    result.append(first_term - second_term - normal_shift)
    for i in range(0, spatial_velocity.size + 2):
        result.append(lapse * normal_velocity - normal_shift)
    result.append(first_term + second_term - normal_shift)
    result.append(lapse - normal_shift)
    return result


# End functions for testing Characteristics.cpp


# Functions for testing ConservativeFromPrimitive.cpp
def tilde_d(rest_mass_density, specific_internal_energy, specific_enthalpy,
            pressure, spatial_velocity, lorentz_factor, magnetic_field,
            sqrt_det_spatial_metric, spatial_metric,
            divergence_cleaning_field):
    return lorentz_factor * rest_mass_density * sqrt_det_spatial_metric


def tilde_tau(rest_mass_density, specific_internal_energy, specific_enthalpy,
              pressure, spatial_velocity, lorentz_factor, magnetic_field,
              sqrt_det_spatial_metric, spatial_metric,
              divergence_cleaning_field):
    spatial_velocity_squared = vsq(spatial_velocity, spatial_metric)
    return (((
        (pressure * spatial_velocity_squared +
         (lorentz_factor / (1.0 + lorentz_factor) * spatial_velocity_squared +
          specific_internal_energy) * rest_mass_density) * lorentz_factor**2) +
             0.5 * b_squared(magnetic_field, spatial_metric) *
             (1.0 + spatial_velocity_squared) - 0.5 * np.square(
                 b_dot_v(magnetic_field, spatial_velocity, spatial_metric))) *
            sqrt_det_spatial_metric)


def tilde_s(rest_mass_density, specific_internal_energy, specific_enthalpy,
            pressure, spatial_velocity, lorentz_factor, magnetic_field,
            sqrt_det_spatial_metric, spatial_metric,
            divergence_cleaning_field):
    return ((spatial_velocity_one_form(spatial_velocity, spatial_metric) *
             (lorentz_factor**2 * specific_enthalpy * rest_mass_density +
              b_squared(magnetic_field, spatial_metric)) -
             magnetic_field_one_form(magnetic_field, spatial_metric) *
             b_dot_v(magnetic_field, spatial_velocity, spatial_metric)) *
            sqrt_det_spatial_metric)


def tilde_b(rest_mass_density, specific_internal_energy, specific_enthalpy,
            pressure, spatial_velocity, lorentz_factor, magnetic_field,
            sqrt_det_spatial_metric, spatial_metric,
            divergence_cleaning_field):
    return sqrt_det_spatial_metric * magnetic_field


def tilde_phi(rest_mass_density, specific_internal_energy, specific_enthalpy,
              pressure, spatial_velocity, lorentz_factor, magnetic_field,
              sqrt_det_spatial_metric, spatial_metric,
              divergence_cleaning_field):
    return sqrt_det_spatial_metric * divergence_cleaning_field


# End functions for testing ConservativeFromPrimitive.cpp
