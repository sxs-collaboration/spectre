# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def b_dot_v(magnetic_field, spatial_velocity, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(magnetic_field, spatial_velocity))


def b_squared(magnetic_field, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(magnetic_field, magnetic_field))


def p_star(pressure, b_dot_v, b_squared, lorentz_factor):
    return pressure + 0.5 * (b_dot_v**2 + b_squared / lorentz_factor**2)


def stress_tensor(spatial_velocity, magnetic_field, rest_mass_density,
                  specific_enthalpy, lorentz_factor, pressure, spatial_metric,
                  inv_spatial_metric, sqrt_det_spatial_metric):
    b_squared_ = b_squared(magnetic_field, spatial_metric)
    b_dot_v_ = b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    return (sqrt_det_spatial_metric *
            ((specific_enthalpy * rest_mass_density * lorentz_factor**2 +
              b_squared_) * np.outer(spatial_velocity, spatial_velocity) +
             p_star(pressure, b_dot_v_, b_squared_, lorentz_factor) *
             inv_spatial_metric - b_dot_v_ *
             (np.outer(magnetic_field, spatial_velocity) +
              np.outer(spatial_velocity, magnetic_field)) -
             np.outer(magnetic_field, magnetic_field) / lorentz_factor**2))


def source_tilde_tau(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                     spatial_velocity, magnetic_field, rest_mass_density,
                     specific_enthalpy, lorentz_factor, pressure, lapse,
                     d_lapse, d_shift, spatial_metric, d_spatial_metric,
                     inv_spatial_metric, sqrt_det_spatial_metric,
                     extrinsic_curvature, constraint_damping_parameter):
    stress_tensor_ = stress_tensor(spatial_velocity, magnetic_field,
                                   rest_mass_density, specific_enthalpy,
                                   lorentz_factor, pressure, spatial_metric,
                                   inv_spatial_metric, sqrt_det_spatial_metric)
    return (
        lapse * np.einsum("ab, ab", stress_tensor_, extrinsic_curvature) -
        np.einsum("ab, ab", inv_spatial_metric, np.outer(tilde_s, d_lapse)))


def source_tilde_s(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                   spatial_velocity, magnetic_field, rest_mass_density,
                   specific_enthalpy, lorentz_factor, pressure, lapse, d_lapse,
                   d_shift, spatial_metric, d_spatial_metric,
                   inv_spatial_metric, sqrt_det_spatial_metric,
                   extrinsic_curvature, constraint_damping_parameter):
    stress_tensor_ = stress_tensor(spatial_velocity, magnetic_field,
                                   rest_mass_density, specific_enthalpy,
                                   lorentz_factor, pressure, spatial_metric,
                                   inv_spatial_metric, sqrt_det_spatial_metric)
    return (
        0.5 * lapse * np.einsum("ab, iab", stress_tensor_, d_spatial_metric) +
        np.einsum("a, ia", tilde_s, d_shift) - (tilde_d + tilde_tau) * d_lapse)


def source_tilde_b(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                   spatial_velocity, magnetic_field, rest_mass_density,
                   specific_enthalpy, lorentz_factor, pressure, lapse, d_lapse,
                   d_shift, spatial_metric, d_spatial_metric,
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
                     specific_enthalpy, lorentz_factor, pressure, lapse,
                     d_lapse, d_shift, spatial_metric, d_spatial_metric,
                     inv_spatial_metric, sqrt_det_spatial_metric,
                     extrinsic_curvature, constraint_damping_parameter):
    return (lapse * tilde_phi *
            (-np.einsum("ab, ab", inv_spatial_metric, extrinsic_curvature) -
             constraint_damping_parameter) +
            np.einsum("a, a", tilde_b, d_lapse))
