# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from PointwiseFunctions.GeneralRelativity.ComputeGhQuantities import (
    spacetime_deriv_detg, deriv_lapse, dt_lapse)


def spatial_weight_function(coords, r_max):
    r2 = np.sum([coords[i]**2 for i in range(len(coords))])
    return np.exp(-r2 / r_max / r_max)


def spacetime_deriv_spatial_weight_function(coords, r_max, W=None):
    if W is None:
        W = spatial_weight_function(coords, r_max)
    DW = np.zeros(len(coords) + 1)
    DW[1:] = -2. * W / r_max / r_max * coords
    return DW


def log_fac(lapse, sqrt_det_spatial_metric, exponent):
    # if exponent == 0, the first term automatically vanishes
    return exponent * np.log(sqrt_det_spatial_metric**2) - np.log(lapse)


def spacetime_deriv_log_fac(lapse, shift, spacetime_unit_normal,
                            inverse_spatial_metric, sqrt_det_spatial_metric,
                            dt_spatial_metric, pi, phi, exponent):
    spatial_dim = len(shift)
    detg = sqrt_det_spatial_metric**2
    dg = spacetime_deriv_detg(sqrt_det_spatial_metric, inverse_spatial_metric,
                              dt_spatial_metric, phi)
    d0N = dt_lapse(lapse, shift, spacetime_unit_normal, phi, pi)
    d3N = deriv_lapse(lapse, spacetime_unit_normal, phi)
    d4N = np.zeros(1 + spatial_dim)
    d4N[0] = d0N
    d4N[1:] = d3N
    # if exponent == 0, the first term automatically vanishes
    d_logfac = (exponent / detg) * dg - (1. / lapse) * d4N
    return d_logfac


def spacetime_deriv_pow_log_fac(lapse, shift, spacetime_unit_normal,
                                inverse_spatial_metric,
                                sqrt_det_spatial_metric, dt_spatial_metric, pi,
                                phi, g_exponent, exponent):
    exponent = int(exponent)
    dlogfac = spacetime_deriv_log_fac(lapse, shift, spacetime_unit_normal,
                                      inverse_spatial_metric,
                                      sqrt_det_spatial_metric,
                                      dt_spatial_metric, pi, phi, g_exponent)
    logfac = log_fac(lapse, sqrt_det_spatial_metric, g_exponent)
    return exponent * np.power(logfac, exponent - 1) * dlogfac
