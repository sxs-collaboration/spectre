# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def b_dot_v(magnetic_field, spatial_velocity, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(magnetic_field, spatial_velocity))


def bsq(magnetic_field, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(magnetic_field, magnetic_field))


def magnetic_field_one_form(magnetic_field, spatial_metric):
    return np.einsum("a, ia", magnetic_field, spatial_metric)


def p_star(pressure, b_dot_v, bsq, lorentz_factor):
    return pressure + 0.5 * (b_dot_v**2 + bsq / lorentz_factor**2)


def spatial_velocity_one_form(spatial_velocity, spatial_metric):
    return np.einsum("a, ia", spatial_velocity, spatial_metric)


def stress_tensor(spatial_velocity, magnetic_field, rest_mass_density,
                  specific_enthalpy, lorentz_factor, pressure,
                  spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric):
    bsq_ = bsq(magnetic_field, spatial_metric)
    b_dot_v_ = b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    return (sqrt_det_spatial_metric *
            ((specific_enthalpy * rest_mass_density * lorentz_factor**2 +
              bsq_) *
             np.outer(spatial_velocity, spatial_velocity) +
             p_star(pressure, b_dot_v_, bsq_, lorentz_factor) *
             inv_spatial_metric -
             b_dot_v_ * (np.outer(magnetic_field, spatial_velocity) +
                         np.outer(spatial_velocity, magnetic_field)) -
             np.outer(magnetic_field, magnetic_field) / lorentz_factor**2))


def vsq(spatial_velocity, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(spatial_velocity, spatial_velocity))


# Functions for testing Characteristics.cpp
def characteristic_speeds(lapse, shift, spatial_velocity, spatial_velocity_sqrd,
                          sound_speed_sqrd, alfven_speed_sqrd, normal_oneform):
    normal_velocity = np.dot(spatial_velocity, normal_oneform)
    normal_shift = np.dot(shift, normal_oneform)
    sound_speed_sqrd += alfven_speed_sqrd * (1.0 - sound_speed_sqrd)
    prefactor = lapse / (1.0 - spatial_velocity_sqrd * sound_speed_sqrd)
    first_term = prefactor * normal_velocity * (1.0 - sound_speed_sqrd)
    second_term = (prefactor * np.sqrt(sound_speed_sqrd) *
                   np.sqrt((1.0 - spatial_velocity_sqrd) *
                           (1.0 - spatial_velocity_sqrd * sound_speed_sqrd
                            - normal_velocity * normal_velocity *
                            (1.0 - sound_speed_sqrd))))
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
            sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field):
    return lorentz_factor * rest_mass_density * sqrt_det_spatial_metric


def tilde_tau(rest_mass_density, specific_internal_energy, specific_enthalpy,
              pressure, spatial_velocity, lorentz_factor, magnetic_field,
              sqrt_det_spatial_metric, spatial_metric,
              divergence_cleaning_field):
    spatial_velocity_squared = vsq(spatial_velocity, spatial_metric)
    return ((((pressure * spatial_velocity_squared
             + (lorentz_factor/(1.0 + lorentz_factor) * spatial_velocity_squared
                + specific_internal_energy) * rest_mass_density)
               * lorentz_factor**2) +
             0.5 * bsq(magnetic_field, spatial_metric) *
             (1.0 + spatial_velocity_squared) -
             0.5 * np.square(b_dot_v(magnetic_field, spatial_velocity,
               spatial_metric))) * sqrt_det_spatial_metric)


def tilde_s(rest_mass_density, specific_internal_energy, specific_enthalpy,
            pressure, spatial_velocity, lorentz_factor, magnetic_field,
            sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field):
    return ((spatial_velocity_one_form(spatial_velocity, spatial_metric) *
             (lorentz_factor**2 * specific_enthalpy * rest_mass_density + bsq(
                 magnetic_field, spatial_metric)) -
             magnetic_field_one_form(magnetic_field, spatial_metric) * b_dot_v(
                 magnetic_field, spatial_velocity, spatial_metric)) *
            sqrt_det_spatial_metric)


def tilde_b(rest_mass_density, specific_internal_energy, specific_enthalpy,
            pressure, spatial_velocity, lorentz_factor, magnetic_field,
            sqrt_det_spatial_metric, spatial_metric, divergence_cleaning_field):
    return sqrt_det_spatial_metric * magnetic_field


def tilde_phi(rest_mass_density, specific_internal_energy, specific_enthalpy,
              pressure, spatial_velocity, lorentz_factor, magnetic_field,
              sqrt_det_spatial_metric, spatial_metric,
              divergence_cleaning_field):
    return sqrt_det_spatial_metric * divergence_cleaning_field


# End functions for testing ConservativeFromPrimitive.cpp


# Functions for testing Fluxes.cpp
def tilde_d_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 pressure, spatial_velocity, lorentz_factor, magnetic_field):
    return tilde_d * (lapse * spatial_velocity - shift)


def tilde_tau_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, pressure, spatial_velocity,
                   lorentz_factor, magnetic_field):
    b_dot_v_ = b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    return (sqrt_det_spatial_metric * lapse * p_star(
        pressure, b_dot_v_, bsq(magnetic_field, spatial_metric),
        lorentz_factor) * spatial_velocity + tilde_tau *
            (lapse * spatial_velocity - shift) - lapse * b_dot_v_ * tilde_b)


def tilde_s_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 pressure, spatial_velocity, lorentz_factor, magnetic_field):
    b_dot_v_ = b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    b_i = (magnetic_field_one_form(magnetic_field, spatial_metric)
           / lorentz_factor + spatial_velocity_one_form(
               spatial_velocity, spatial_metric) * lorentz_factor * b_dot_v_)
    result = np.outer(lapse * spatial_velocity - shift, tilde_s)
    result -= lapse / lorentz_factor * np.outer(tilde_b, b_i)
    result += (sqrt_det_spatial_metric * lapse * p_star(
        pressure, b_dot_v_, bsq(magnetic_field, spatial_metric),
        lorentz_factor) * np.identity(shift.size))
    return result


def tilde_b_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 pressure, spatial_velocity, lorentz_factor, magnetic_field):
    result = np.outer(lapse * spatial_velocity - shift, tilde_b)
    result += lapse * inv_spatial_metric * tilde_phi
    result -= lapse * np.outer(tilde_b, spatial_velocity)
    return result


def tilde_phi_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, pressure, spatial_velocity,
                   lorentz_factor, magnetic_field):
    return lapse * tilde_b - tilde_phi * shift


# End functions for testing Fluxes.cpp


# Functions for testing Sources.cpp
def source_tilde_tau(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                     spatial_velocity, magnetic_field, rest_mass_density,
                     specific_enthalpy, lorentz_factor, pressure,
                     lapse, d_lapse, d_shift, spatial_metric, d_spatial_metric,
                     inv_spatial_metric, sqrt_det_spatial_metric,
                     extrinsic_curvature, constraint_damping_parameter):
    stress_tensor_ = stress_tensor(spatial_velocity, magnetic_field,
                                   rest_mass_density, specific_enthalpy,
                                   lorentz_factor, pressure, spatial_metric,
                                   inv_spatial_metric, sqrt_det_spatial_metric)
    return (lapse * np.einsum("ab, ab", stress_tensor_, extrinsic_curvature)
            - np.einsum("ab, ab", inv_spatial_metric,
                        np.outer(tilde_s, d_lapse)))


def source_tilde_s(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                   spatial_velocity, magnetic_field, rest_mass_density,
                   specific_enthalpy, lorentz_factor, pressure,
                   lapse, d_lapse, d_shift, spatial_metric, d_spatial_metric,
                   inv_spatial_metric, sqrt_det_spatial_metric,
                   extrinsic_curvature, constraint_damping_parameter):
    stress_tensor_ = stress_tensor(spatial_velocity, magnetic_field,
                                   rest_mass_density, specific_enthalpy,
                                   lorentz_factor, pressure, spatial_metric,
                                   inv_spatial_metric, sqrt_det_spatial_metric)
    return (0.5 * lapse *
            np.einsum("ab, iab", stress_tensor_, d_spatial_metric) +
            np.einsum("a, ia", tilde_s, d_shift) -
            (tilde_d + tilde_tau) * d_lapse)


def source_tilde_b(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                   spatial_velocity, magnetic_field, rest_mass_density,
                   specific_enthalpy, lorentz_factor, pressure,
                   lapse, d_lapse, d_shift, spatial_metric, d_spatial_metric,
                   inv_spatial_metric, sqrt_det_spatial_metric,
                   extrinsic_curvature, constraint_damping_parameter):
    term_one = np.einsum("ab, iab", inv_spatial_metric, d_spatial_metric)
    term_two = np.einsum("ab, aib", inv_spatial_metric, d_spatial_metric)
    return (tilde_phi * np.einsum("a, ia", d_lapse, inv_spatial_metric) -
            np.einsum("a, ai", tilde_b, d_shift) + lapse * tilde_phi *
            (0.5 * np.einsum("ia, a", inv_spatial_metric, term_one) -
             np.einsum("ia, a", inv_spatial_metric, term_two)))


def source_tilde_phi(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                     spatial_velocity, magnetic_field, rest_mass_density,
                     specific_enthalpy, lorentz_factor, pressure,
                     lapse, d_lapse, d_shift, spatial_metric, d_spatial_metric,
                     inv_spatial_metric, sqrt_det_spatial_metric,
                     extrinsic_curvature, constraint_damping_parameter):
    return (lapse * tilde_phi *
            (-np.einsum("ab, ab", inv_spatial_metric, extrinsic_curvature) -
             constraint_damping_parameter) +
            np.einsum("a, a", tilde_b, d_lapse))


# End functions for testing Sources.cpp
