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


def tilde_d_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 pressure, spatial_velocity, lorentz_factor, magnetic_field):
    return tilde_d * (lapse * spatial_velocity - shift)


def tilde_tau_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, pressure, spatial_velocity,
                   lorentz_factor, magnetic_field):
    b_dot_v_ = b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    return (
        sqrt_det_spatial_metric * lapse *
        p_star(pressure, b_dot_v_, b_squared(magnetic_field, spatial_metric),
               lorentz_factor) * spatial_velocity + tilde_tau *
        (lapse * spatial_velocity - shift) - lapse * b_dot_v_ * tilde_b)


def tilde_s_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 pressure, spatial_velocity, lorentz_factor, magnetic_field):
    b_dot_v_ = b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    b_i = (magnetic_field_one_form(magnetic_field, spatial_metric) /
           lorentz_factor +
           spatial_velocity_one_form(spatial_velocity, spatial_metric) *
           lorentz_factor * b_dot_v_)
    result = np.outer(lapse * spatial_velocity - shift, tilde_s)
    result -= lapse / lorentz_factor * np.outer(tilde_b, b_i)
    result += (
        sqrt_det_spatial_metric * lapse *
        p_star(pressure, b_dot_v_, b_squared(magnetic_field, spatial_metric),
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
