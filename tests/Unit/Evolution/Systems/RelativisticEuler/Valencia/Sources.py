# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def source_tilde_tau(tilde_d, tilde_tau, tilde_s, spatial_velocity, pressure,
                     lapse, d_lapse, d_shift, d_spatial_metric,
                     inv_spatial_metric, sqrt_det_spatial_metric,
                     extrinsic_curvature):
    upper_tilde_s = np.einsum("a, ia", tilde_s, inv_spatial_metric)
    densitized_stress = (
        0.5 * np.outer(upper_tilde_s, spatial_velocity) +
        0.5 * np.outer(spatial_velocity, upper_tilde_s) +
        sqrt_det_spatial_metric * pressure * inv_spatial_metric)
    return (
        lapse * np.einsum("ab, ab", densitized_stress, extrinsic_curvature) -
        np.einsum("ab, ab", inv_spatial_metric, np.outer(tilde_s, d_lapse)))


def source_tilde_s(tilde_d, tilde_tau, tilde_s, spatial_velocity, pressure,
                   lapse, d_lapse, d_shift, d_spatial_metric,
                   inv_spatial_metric, sqrt_det_spatial_metric,
                   extrinsic_curvature):
    upper_tilde_s = np.einsum("a, ia", tilde_s, inv_spatial_metric)
    densitized_stress = (
        np.outer(upper_tilde_s, spatial_velocity) +
        sqrt_det_spatial_metric * pressure * inv_spatial_metric)
    return (np.einsum("a, ia", tilde_s, d_shift) - d_lapse *
            (tilde_tau + tilde_d) + 0.5 * lapse *
            np.einsum("ab, iab", densitized_stress, d_spatial_metric))
