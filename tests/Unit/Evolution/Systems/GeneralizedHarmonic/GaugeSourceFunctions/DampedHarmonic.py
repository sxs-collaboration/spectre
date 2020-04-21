# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from PointwiseFunctions.GeneralRelativity.ComputeGhQuantities import (
    spacetime_deriv_detg, deriv_lapse, dt_lapse, deriv_spatial_metric,
    dt_spatial_metric, deriv_shift, dt_shift)
from PointwiseFunctions.GeneralRelativity.ComputeSpacetimeQuantities import (
    derivatives_of_spacetime_metric, inverse_spacetime_metric,
    spacetime_normal_vector)
from .DampedWaveHelpers import (spatial_weight_function,
                                spacetime_deriv_spatial_weight_function,
                                log_fac)


def roll_on_function(time, t_start, sigma_t):
    if time < t_start:
        return 0.
    return 1. - np.exp(-((time - t_start) / sigma_t)**4)


def time_deriv_roll_on_function(time, t_start, sigma_t):
    if time < t_start:
        return 0.
    tnrm = (time - t_start) / sigma_t
    return np.exp(-tnrm**4) * 4 * tnrm**3 / sigma_t


def spatial_metric_from_spacetime_metric(spacetime_metric):
    return spacetime_metric[1:, 1:]


def damped_harmonic_gauge_source_function_rollon(
    gauge_h_init, dgauge_h_init, lapse, shift, unit_normal_one_form,
    sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi, phi,
    time, coords, amp_coef_L1, amp_coef_L2, amp_coef_S, rollon_start_time,
    rollon_width, sigma_r):

    # We cannot pass int through pypp right now, so we hard-code exponents.
    exp_L1 = 4
    exp_L2 = 4
    exp_S = 4

    log_sqrtg_over_lapse = log_fac(lapse, sqrt_det_spatial_metric, 0.5)
    log_one_over_lapse = log_fac(lapse, sqrt_det_spatial_metric, 0.)

    if gauge_h_init is None:
        R = 1.
    else:
        R = roll_on_function(time, rollon_start_time, rollon_width)
    W = spatial_weight_function(coords, sigma_r)

    muL1 = amp_coef_L1 * R * W * log_sqrtg_over_lapse**exp_L1
    muL2 = amp_coef_L2 * R * W * log_one_over_lapse**exp_L2
    muS = amp_coef_S * R * W * log_sqrtg_over_lapse**exp_S
    h_pre1 = muL1 * log_sqrtg_over_lapse + muL2 * log_one_over_lapse
    h_pre2 = muS / lapse
    if gauge_h_init is None:
        init_gauge_term = 0.
    else:
        init_gauge_term = (1. - R) * gauge_h_init

    return (init_gauge_term + h_pre1 * unit_normal_one_form -
            h_pre2 * np.einsum('ai,i->a', spacetime_metric[:, 1:], shift))


def spacetime_deriv_damped_harmonic_gauge_source_function_rollon(
    gauge_h_init, dgauge_h_init, lapse, shift, spacetime_unit_normal_one_form,
    sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi, phi,
    time, coords, amp_coef_L1, amp_coef_L2, amp_coef_S, rollon_start_time,
    rollon_width, sigma_r):
    spatial_dim = len(shift)

    spacetime_unit_normal = spacetime_normal_vector(lapse, shift)
    spatial_metric = spatial_metric_from_spacetime_metric(spacetime_metric)
    det_spatial_metric = sqrt_det_spatial_metric**2
    d3_spatial_metric = deriv_spatial_metric(phi)
    inv_spacetime_metric = inverse_spacetime_metric(lapse, shift,
                                                    inverse_spatial_metric)
    d0_spatial_metric = dt_spatial_metric(lapse, shift, phi, pi)

    log_sqrtg_over_lapse = np.log(sqrt_det_spatial_metric / lapse)
    log_sqrtg_over_lapse_pow3 = log_sqrtg_over_lapse**3
    log_sqrtg_over_lapse_pow4 = log_sqrtg_over_lapse * log_sqrtg_over_lapse_pow3
    log_sqrtg_over_lapse_pow5 = log_sqrtg_over_lapse * log_sqrtg_over_lapse_pow4
    one_over_lapse = 1. / lapse
    log_one_over_lapse = np.log(one_over_lapse)
    log_one_over_lapse_pow4 = log_one_over_lapse**4
    log_one_over_lapse_pow5 = log_one_over_lapse * log_one_over_lapse_pow4

    if gauge_h_init is None:
        R = 1.
    else:
        R = roll_on_function(time, rollon_start_time, rollon_width)
    W = spatial_weight_function(coords, sigma_r)

    muL1 = R * W * log_sqrtg_over_lapse_pow4
    muS_over_N = muL1 / lapse

    mu1 = muL1 * log_sqrtg_over_lapse
    mu2 = R * W * log_one_over_lapse_pow5

    d4_W = spacetime_deriv_spatial_weight_function(coords, sigma_r)
    if gauge_h_init is None:
        d0_R = 0.
    else:
        d0_R = time_deriv_roll_on_function(time, rollon_start_time,
                                           rollon_width)
    d4_RW = R * d4_W
    d4_RW[0] += W * d0_R

    d4_g = spacetime_deriv_detg(sqrt_det_spatial_metric,
                                inverse_spatial_metric, d0_spatial_metric, phi)

    d0_N = dt_lapse(lapse, shift, spacetime_unit_normal, phi, pi)
    d3_N = deriv_lapse(lapse, spacetime_unit_normal, phi)
    d4_N = np.zeros(1 + spatial_dim)
    d4_N[0] = d0_N
    d4_N[1:] = d3_N

    d0_shift = dt_shift(lapse, shift, inverse_spatial_metric,
                        spacetime_unit_normal, phi, pi)
    d3_shift = deriv_shift(lapse, inv_spacetime_metric, spacetime_unit_normal,
                           phi)
    d4_shift = np.einsum('a,b->ab', np.zeros(spatial_dim + 1),
                         np.zeros(spatial_dim + 1))
    d4_shift[0, 1:] = d0_shift
    d4_shift[1:, 1:] = d3_shift

    prefac0 = 2. * R * W * log_sqrtg_over_lapse_pow3
    prefac1 = 2.5 * R * W * log_sqrtg_over_lapse_pow4
    prefac2 = -5. * R * W * log_one_over_lapse_pow4 * one_over_lapse
    d4_muL1 = log_sqrtg_over_lapse_pow4 * d4_RW + prefac0 * (
        d4_g / det_spatial_metric - 2. * one_over_lapse * d4_N)
    d4_mu1 = log_sqrtg_over_lapse_pow5 * d4_RW + prefac1 * (
        d4_g / det_spatial_metric - 2. * one_over_lapse * d4_N)
    d4_mu2 = log_one_over_lapse_pow5 * d4_RW + prefac2 * d4_N

    d4_normal_one_form = np.einsum('a,b->ab', np.zeros(spatial_dim + 1),
                                   np.zeros(spatial_dim + 1))
    d4_normal_one_form[:, 0] = -d4_N

    d4_muS_over_N = one_over_lapse * d4_muL1 - muL1 * one_over_lapse**2 * d4_N

    d4_psi = derivatives_of_spacetime_metric(lapse, d0_N, d3_N, shift,
                                             d0_shift, d3_shift,
                                             spatial_metric, d0_spatial_metric,
                                             d3_spatial_metric)

    if gauge_h_init is None:
        dT1 = 0.
    else:
        dT1 = (1. - R) * dgauge_h_init
        dT1[0, :] -= d0_R * gauge_h_init

    dT2 = (amp_coef_L1 * mu1 +
           amp_coef_L2 * mu2) * d4_normal_one_form + np.einsum(
               'a,b->ab', amp_coef_L1 * d4_mu1 + amp_coef_L2 * d4_mu2,
               spacetime_unit_normal_one_form)

    dT3 = np.einsum('a,b->ab', np.zeros(spatial_dim + 1),
                    np.zeros(spatial_dim + 1))
    dT3 -= np.einsum('a,b->ab', d4_muS_over_N,
                     np.einsum('bi,i->b', spacetime_metric[:, 1:], shift))
    dT3 -= muS_over_N * np.einsum('abi,i->ab', d4_psi[:, :, 1:], shift)
    dT3 -= muS_over_N * np.einsum('ai,bi->ab', d4_shift[:, 1:],
                                  spacetime_metric[:, 1:])
    return dT1 + dT2 + amp_coef_S * dT3


def damped_harmonic_gauge_source_function(lapse, shift, unit_normal_one_form,
                                          sqrt_det_spatial_metric,
                                          inverse_spatial_metric,
                                          spacetime_metric, pi, phi, coords,
                                          amp_coef_L1, amp_coef_L2, amp_coef_S,
                                          sigma_r):
    return damped_harmonic_gauge_source_function_rollon(
        None, None, lapse, shift, unit_normal_one_form,
        sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
        phi, None, coords, amp_coef_L1, amp_coef_L2, amp_coef_S, None, None,
        sigma_r)


def spacetime_deriv_damped_harmonic_gauge_source_function(
    lapse, shift, spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
    inverse_spatial_metric, spacetime_metric, pi, phi, coords, amp_coef_L1,
    amp_coef_L2, amp_coef_S, sigma_r):

    return spacetime_deriv_damped_harmonic_gauge_source_function_rollon(
        None, None, lapse, shift, spacetime_unit_normal_one_form,
        sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
        phi, None, coords, amp_coef_L1, amp_coef_L2, amp_coef_S, None, None,
        sigma_r)
