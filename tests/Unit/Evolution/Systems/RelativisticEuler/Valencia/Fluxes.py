# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def tilde_d_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                 sqrt_det_spatial_metric, pressure, spatial_velocity):
    return tilde_d * (lapse * spatial_velocity - shift)


def tilde_tau_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                   sqrt_det_spatial_metric, pressure, spatial_velocity):
    return (sqrt_det_spatial_metric * lapse * pressure * spatial_velocity +
            tilde_tau * (lapse * spatial_velocity - shift))


def tilde_s_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                 sqrt_det_spatial_metric, pressure, spatial_velocity):
    result = np.outer(lapse * spatial_velocity - shift, tilde_s)
    result += (sqrt_det_spatial_metric * lapse * pressure *
               np.identity(shift.size))
    return result
