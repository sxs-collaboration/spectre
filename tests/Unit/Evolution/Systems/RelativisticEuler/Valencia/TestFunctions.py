# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing Characteristics.cpp
def characteristic_speeds(lapse, shift, spatial_velocity, spatial_velocity_sqrd,
                          sound_speed_sqrd, normal_oneform):
    normal_velocity = np.dot(spatial_velocity, normal_oneform)
    normal_shift = np.dot(shift, normal_oneform)
    prefactor = lapse / (1.0 - spatial_velocity_sqrd * sound_speed_sqrd)
    first_term = prefactor * normal_velocity * (1.0 - sound_speed_sqrd)
    second_term = (prefactor * np.sqrt(sound_speed_sqrd) *
                   np.sqrt((1.0 - spatial_velocity_sqrd) *
                           (1.0 - spatial_velocity_sqrd * sound_speed_sqrd
                            - normal_velocity * normal_velocity *
                            (1.0 - sound_speed_sqrd))))
    result = [first_term - second_term - normal_shift]
    for i in range(0, spatial_velocity.size):
        result.append(lapse * normal_velocity - normal_shift)
    result.append(first_term + second_term - normal_shift)
    return result


# End functions for testing Characteristics.cpp


# Functions for testing ConservativeFromPrimitive.cpp
def tilde_d(rest_mass_density, specific_internal_energy,
            spatial_velocity_oneform, spatial_velocity_squared, lorentz_factor,
            specific_enthalpy, pressure, sqrt_det_spatial_metric):
    return lorentz_factor * rest_mass_density * sqrt_det_spatial_metric


def tilde_tau(rest_mass_density, specific_internal_energy,
              spatial_velocity_oneform, spatial_velocity_squared,
              lorentz_factor, specific_enthalpy, pressure,
              sqrt_det_spatial_metric):
    return ((pressure * spatial_velocity_squared
             + (lorentz_factor/(1.0 + lorentz_factor) * spatial_velocity_squared
                + specific_internal_energy) * rest_mass_density)
            * lorentz_factor**2 * sqrt_det_spatial_metric)


def tilde_s(rest_mass_density, specific_internal_energy,
            spatial_velocity_oneform, spatial_velocity_squared, lorentz_factor,
            specific_enthalpy, pressure, sqrt_det_spatial_metric):
    return (spatial_velocity_oneform * lorentz_factor**2 * specific_enthalpy
            * rest_mass_density * sqrt_det_spatial_metric)


# End functions for testing ConservativeFromPrimitive.cpp


# Functions for testing Equations.cpp
def source_tilde_d(tilde_d, tilde_tau, tilde_s, spatial_velocity, pressure,
                   lapse, d_lapse, d_shift, d_spatial_metric,
                   inv_spatial_metric, sqrt_det_spatial_metric,
                   extrinsic_curvature):
    return 0.0


def source_tilde_tau(tilde_d, tilde_tau, tilde_s, spatial_velocity, pressure,
                     lapse, d_lapse, d_shift, d_spatial_metric,
                     inv_spatial_metric, sqrt_det_spatial_metric,
                     extrinsic_curvature):
    upper_tilde_s = np.einsum("a, ia", tilde_s, inv_spatial_metric)
    densitized_stress = (0.5 * np.outer(upper_tilde_s, spatial_velocity)
                         + 0.5 * np.outer(spatial_velocity, upper_tilde_s)
                         + sqrt_det_spatial_metric * pressure
                         * inv_spatial_metric)
    return (lapse * np.einsum("ab, ab", densitized_stress, extrinsic_curvature)
            - np.einsum("ab, ab", inv_spatial_metric,
                        np.outer(tilde_s, d_lapse)))


def source_tilde_s(tilde_d, tilde_tau, tilde_s, spatial_velocity, pressure,
                   lapse, d_lapse, d_shift, d_spatial_metric,
                   inv_spatial_metric, sqrt_det_spatial_metric,
                   extrinsic_curvature):
    upper_tilde_s = np.einsum("a, ia", tilde_s, inv_spatial_metric)
    densitized_stress = (np.outer(upper_tilde_s, spatial_velocity)
                         + sqrt_det_spatial_metric * pressure
                         * inv_spatial_metric)
    return (np.einsum("a, ia", tilde_s, d_shift)
            - d_lapse * (tilde_tau + tilde_d)
            + 0.5 * lapse * np.einsum("ab, iab", densitized_stress,
                                      d_spatial_metric))


# End functions for testing Equations.cpp


# Functions for testing Fluxes.cpp
def tilde_d_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                 sqrt_det_spatial_metric, pressure, spatial_velocity):
    return tilde_d * (lapse * spatial_velocity - shift)


def tilde_tau_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                   sqrt_det_spatial_metric, pressure, spatial_velocity):
    return (sqrt_det_spatial_metric * lapse * pressure * spatial_velocity
            + tilde_tau * (lapse * spatial_velocity - shift))


def tilde_s_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                 sqrt_det_spatial_metric, pressure, spatial_velocity):
    result = np.outer(lapse * spatial_velocity - shift, tilde_s)
    result += (sqrt_det_spatial_metric * lapse * pressure
               * np.identity(shift.size))
    return result


# End functions for testing Fluxes.cpp
