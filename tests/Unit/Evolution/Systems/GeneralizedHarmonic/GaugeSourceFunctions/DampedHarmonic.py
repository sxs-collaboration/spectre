# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import os
import sys
from PointwiseFunctions.GeneralRelativity.ComputeGhQuantities import (
    spacetime_deriv_detg, deriv_lapse, dt_lapse, deriv_spatial_metric,
    dt_spatial_metric, deriv_shift, dt_shift
)
from PointwiseFunctions.GeneralRelativity.ComputeSpacetimeQuantities import (
    derivatives_of_spacetime_metric, inverse_spacetime_metric,
    spacetime_normal_vector
)


def weight_function(coords, r_max):
    r2 = np.sum([coords[i]**2 for i in range(len(coords))])
    return np.exp(- r2 / r_max / r_max)


def spacetime_deriv_weight_function(coords, r_max):
    W = weight_function(coords, r_max)
    DW = np.zeros(len(coords) + 1)
    DW[1:] = -2. * W / r_max / r_max * coords
    return DW


def roll_on_function(time, t_start, sigma_t):
    if time < t_start:
        return 0.
    return 1. - np.exp(- ((time - t_start) / sigma_t)**4)


def time_deriv_roll_on_function(time, t_start, sigma_t):
    if time < t_start:
        return 0.
    tnrm = (time - t_start) / sigma_t
    return np.exp(- tnrm**4) * 4 * tnrm**3 / sigma_t


def log_fac(lapse, sqrt_det_spatial_metric, exponent):
    # if exponent == 0, the first term automatically vanishes
    return exponent * np.log(sqrt_det_spatial_metric**2) - np.log(lapse)


def spacetime_deriv_log_fac(lapse, shift, spacetime_unit_normal,
                            inverse_spatial_metric, sqrt_det_spatial_metric,
                            dt_spatial_metric, pi, phi, exponent):
    spatial_dim = len(shift)
    detg = sqrt_det_spatial_metric**2
    dg = spacetime_deriv_detg(sqrt_det_spatial_metric,
                                   inverse_spatial_metric, dt_spatial_metric,
                                   phi)
    d0N = dt_lapse(lapse, shift, spacetime_unit_normal, phi, pi)
    d3N = deriv_lapse(lapse, spacetime_unit_normal, phi)
    d4N = np.zeros(1 + spatial_dim)
    d4N[0] = d0N
    d4N[1:] = d3N
    # if exponent == 0, the first term automatically vanishes
    d_logfac = (exponent / detg) * dg - (1. / lapse) * d4N
    return d_logfac


def spacetime_deriv_pow_log_fac(lapse, shift, spacetime_unit_normal,
                                inverse_spatial_metric, sqrt_det_spatial_metric,
                                dt_spatial_metric,
                                pi, phi, g_exponent, exponent):
    exponent = int(exponent)
    dlogfac = spacetime_deriv_log_fac(lapse, shift, spacetime_unit_normal,
                                      inverse_spatial_metric,
                                      sqrt_det_spatial_metric,
                                      dt_spatial_metric,
                                      pi, phi, g_exponent)
    logfac = log_fac(lapse, sqrt_det_spatial_metric, g_exponent)
    return exponent * np.power(logfac, exponent - 1) * dlogfac


def spatial_metric_from_spacetime_metric(spacetime_metric):
    return spacetime_metric[1:, 1:]


# In the next two functions, assume that amplitude pre-factors and exponents
# have the following values:
#      amp_coef_{h_init, L1, L2, S} = 1., 1., 1., 1.
#      exp_{L1, L2, S}              = 4, 4, 4
# and that the roll-on function associated with each term has identical config:
#      t_start, sigma_t  for {_h_init, _L1, _L2, _S}.
def damped_harmonic_gauge_source_function(gauge_h_init, lapse, shift,
                                          sqrt_det_spatial_metric,
                                          spacetime_metric, time, t_start,
                                          sigma_t, coords, r_max):
    unit_normal_one_form = np.zeros(1 + len(shift))
    unit_normal_one_form[0] -= lapse
    log_sqrtg_over_lapse = log_fac(lapse, sqrt_det_spatial_metric, 0.5)
    log_one_over_lapse = log_fac(lapse, sqrt_det_spatial_metric, 0.)
    R = roll_on_function(time, t_start, sigma_t)
    W = weight_function(coords, r_max)
    muL1 = R * W * log_sqrtg_over_lapse**4
    muL2 = R * W * log_one_over_lapse**4
    muS = muL1
    h_pre1 = muL1 * log_sqrtg_over_lapse + muL2 * log_one_over_lapse
    h_pre2 = muS / lapse
    return (1. - R) * gauge_h_init + h_pre1 * unit_normal_one_form - h_pre2 * \
        np.einsum('ai,i->a', spacetime_metric[:, 1:], shift)


def spacetime_deriv_damped_harmonic_gauge_source_function(gauge_h_init,
                                                dgauge_h_init, lapse, shift,
                                                spacetime_unit_normal_one_form,
                                                sqrt_det_spatial_metric,
                                                inverse_spatial_metric,
                                                spacetime_metric, pi, phi, time,
                                                t_start, sigma_t, coords,
                                                r_max):
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

    R = roll_on_function(time, t_start, sigma_t)
    W = weight_function(coords, r_max)

    muL1 = R * W * log_sqrtg_over_lapse_pow4
    muS_over_N = muL1 / lapse

    mu1 = muL1 * log_sqrtg_over_lapse
    mu2 = R * W * log_one_over_lapse_pow5

    d4_W = spacetime_deriv_weight_function(coords, r_max)
    d0_R = time_deriv_roll_on_function(time, t_start, sigma_t)
    d4_RW = R * d4_W
    d4_RW[0] += W * d0_R

    d4_g = spacetime_deriv_detg(sqrt_det_spatial_metric, inverse_spatial_metric,
                                d0_spatial_metric, phi)

    d0_N = dt_lapse(lapse, shift, spacetime_unit_normal, phi, pi)
    d3_N = deriv_lapse(lapse, spacetime_unit_normal, phi)
    d4_N = np.zeros(1 + spatial_dim)
    d4_N[0] = d0_N
    d4_N[1:] = d3_N

    d0_shift = dt_shift(lapse, shift, inverse_spatial_metric,
                             spacetime_unit_normal, phi, pi)
    d3_shift = deriv_shift(lapse, inv_spacetime_metric,
                                spacetime_unit_normal, phi)
    d4_shift = np.einsum('a,b->ab',
                         np.zeros(spatial_dim + 1), np.zeros(spatial_dim + 1))
    d4_shift[0, 1:] = d0_shift
    d4_shift[1:, 1:] = d3_shift

    prefac0 = 2. * R * W * log_sqrtg_over_lapse_pow3
    prefac1 = 2.5 * R * W * log_sqrtg_over_lapse_pow4
    prefac2 = -5. * R * W * log_one_over_lapse_pow4 * one_over_lapse
    d4_muL1 = log_sqrtg_over_lapse_pow4 * d4_RW +\
        prefac0 * (d4_g / det_spatial_metric - 2. * one_over_lapse * d4_N)
    d4_mu1 = log_sqrtg_over_lapse_pow5 * d4_RW +\
        prefac1 * (d4_g / det_spatial_metric - 2. * one_over_lapse * d4_N)
    d4_mu2 = log_one_over_lapse_pow5 * d4_RW + prefac2 * d4_N

    d4_normal_one_form = np.einsum('a,b->ab',
                                   np.zeros(spatial_dim + 1),
                                   np.zeros(spatial_dim + 1))
    d4_normal_one_form[:, 0] = -d4_N

    d4_muS_over_N = one_over_lapse * d4_muL1 - muL1 * one_over_lapse**2 * d4_N

    d4_psi = derivatives_of_spacetime_metric(lapse, d0_N, d3_N,
                                             shift, d0_shift, d3_shift,
                                             spatial_metric,
                                             d0_spatial_metric,
                                             d3_spatial_metric)

    dT1 = (1. - R) * dgauge_h_init
    dT1[0, :] -= d0_R * gauge_h_init

    dT2 = (mu1 + mu2) * d4_normal_one_form +\
        np.einsum('a,b->ab', d4_mu1 + d4_mu2, spacetime_unit_normal_one_form)

    dT3 = np.einsum('a,b->ab',
                    np.zeros(spatial_dim + 1), np.zeros(spatial_dim + 1))
    dT3 -= np.einsum('a,b->ab', d4_muS_over_N,
                     np.einsum('bi,i->b', spacetime_metric[:, 1:], shift))
    dT3 -= muS_over_N * np.einsum('abi,i->ab', d4_psi[:, :, 1:], shift)
    dT3 -= muS_over_N * np.einsum('ai,bi->ab', d4_shift[:, 1:],
                                  spacetime_metric[:, 1:])
    return dT1 + dT2 + dT3
