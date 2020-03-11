# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing M1HydroCoupling.cpp
def hydro_coupling_tilde_e(emissivity, absorption_opacity, scattering_opacity,
                           tilde_j, tilde_hn, tilde_hi, fluid_velocity,
                           lorentz_factor, lapse, spatial_metric,
                           sqrt_det_spatial_metric):
    result = (
        lapse * lorentz_factor *
        (sqrt_det_spatial_metric * emissivity - absorption_opacity * tilde_j) +
        lapse * (absorption_opacity + scattering_opacity) * tilde_hn)
    return result


def hydro_coupling_tilde_s(emissivity, absorption_opacity, scattering_opacity,
                           tilde_j, tilde_hn, tilde_hi, fluid_velocity,
                           lorentz_factor, lapse, spatial_metric,
                           sqrt_det_spatial_metric):
    result = (
        lapse * np.einsum("a, ia", fluid_velocity, spatial_metric) *
        lorentz_factor *
        (sqrt_det_spatial_metric * emissivity - absorption_opacity * tilde_j) -
        lapse * (absorption_opacity + scattering_opacity) * tilde_hi)

    return result


# End of functions for testing M1HydroCoupling.cpp
