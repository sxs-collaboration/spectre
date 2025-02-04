# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from M1HydroCoupling import *


# Functions for testing Sources.cpp
def source_tilde_e(
    tilde_e,
    tilde_s,
    tilde_p,
    lapse,
    d_lapse,
    d_shift,
    d_spatial_metric,
    inv_spatial_metric,
    extrinsic_curvature,
    spatial_metric,
    emissivity,
    absorption_opacity,
    scattering_opacity,
    tilde_j,
    tilde_h_normal,
    tilde_h_spatial,
    spatial_velocity,
    lorentz,
    sqrt_det_spatial_metric,
):
    # coupling term with fluid
    source_n = hydro_coupling_tilde_e(
        emissivity,
        absorption_opacity,
        scattering_opacity,
        tilde_j,
        tilde_h_normal,
        tilde_h_spatial,
        spatial_velocity,
        lorentz,
        lapse,
        spatial_metric,
        sqrt_det_spatial_metric,
    )

    result = (
        lapse * np.einsum("ab, ab", tilde_p, extrinsic_curvature)
        - np.einsum("ab, ab", inv_spatial_metric, np.outer(tilde_s, d_lapse))
        + source_n
    )

    return result


def source_tilde_s(
    tilde_e,
    tilde_s,
    tilde_p,
    lapse,
    d_lapse,
    d_shift,
    d_spatial_metric,
    inv_spatial_metric,
    extrinsic_curvature,
    spatial_metric,
    emissivity,
    absorption_opacity,
    scattering_opacity,
    tilde_j,
    tilde_h_normal,
    tilde_h_spatial,
    spatial_velocity,
    lorentz,
    sqrt_det_spatial_metric,
):
    # coupling with fluid
    source_i = hydro_coupling_tilde_s(
        emissivity,
        absorption_opacity,
        scattering_opacity,
        tilde_j,
        tilde_h_normal,
        tilde_h_spatial,
        spatial_velocity,
        lorentz,
        lapse,
        spatial_metric,
        sqrt_det_spatial_metric,
    )

    result = (
        0.5 * lapse * np.einsum("ab, iab", tilde_p, d_spatial_metric)
        + np.einsum("a, ia", tilde_s, d_shift)
        - tilde_e * d_lapse
        + source_i
    )
    return result


# End of functions for testing Sources.cpp
