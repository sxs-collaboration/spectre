# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def spatial_velocity_oneform(spatial_velocity, spatial_metric):
    return np.einsum("a, ia", spatial_velocity, spatial_metric)


def spatial_velocity_squared(spatial_velocity, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(spatial_velocity, spatial_velocity))


# Functions for testing Characteristics.cpp
def characteristic_speeds(lapse, shift, spatial_velocity,
                          spatial_velocity_sqrd, sound_speed_sqrd,
                          normal_oneform):
    normal_velocity = np.dot(spatial_velocity, normal_oneform)
    normal_shift = np.dot(shift, normal_oneform)
    prefactor = lapse / (1.0 - spatial_velocity_sqrd * sound_speed_sqrd)
    first_term = prefactor * normal_velocity * (1.0 - sound_speed_sqrd)
    second_term = (prefactor * np.sqrt(sound_speed_sqrd) * np.sqrt(
        (1.0 - spatial_velocity_sqrd) *
        (1.0 - spatial_velocity_sqrd * sound_speed_sqrd -
         normal_velocity * normal_velocity * (1.0 - sound_speed_sqrd))))
    result = [first_term - second_term - normal_shift]
    for i in range(0, spatial_velocity.size):
        result.append(lapse * normal_velocity - normal_shift)
    result.append(first_term + second_term - normal_shift)
    return result


# End functions for testing Characteristics.cpp


# Functions for testing ConservativeFromPrimitive.cpp
def tilde_d(rest_mass_density, specific_internal_energy, specific_enthalpy,
            pressure, spatial_velocity, lorentz_factor,
            sqrt_det_spatial_metric, spatial_metric):
    return lorentz_factor * rest_mass_density * sqrt_det_spatial_metric


def tilde_tau(rest_mass_density, specific_internal_energy, specific_enthalpy,
              pressure, spatial_velocity, lorentz_factor,
              sqrt_det_spatial_metric, spatial_metric):
    v_squared = spatial_velocity_squared(spatial_velocity, spatial_metric)
    return ((pressure * v_squared +
             (lorentz_factor /
              (1.0 + lorentz_factor) * v_squared + specific_internal_energy) *
             rest_mass_density) * lorentz_factor**2 * sqrt_det_spatial_metric)


def tilde_s(rest_mass_density, specific_internal_energy, specific_enthalpy,
            pressure, spatial_velocity, lorentz_factor,
            sqrt_det_spatial_metric, spatial_metric):
    return (spatial_velocity_oneform(spatial_velocity, spatial_metric) *
            lorentz_factor**2 * specific_enthalpy * rest_mass_density *
            sqrt_det_spatial_metric)


# End functions for testing ConservativeFromPrimitive.cpp
