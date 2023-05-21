# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from ElectricCurrentDensity import tilde_j_drift


def trace_spatial_Gamma_second_kind(inv_spatial_metric, d_spatial_metric):
    # returns \gamma^{jk} \Gamma^i_{jk}
    term_one = np.einsum("ab, iab", inv_spatial_metric, d_spatial_metric)
    term_two = np.einsum("ab, aib", inv_spatial_metric, d_spatial_metric)
    return -0.5 * np.einsum("ia, a", inv_spatial_metric, term_one) + np.einsum(
        "ia, a", inv_spatial_metric, term_two
    )


def source_tilde_e(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    kappa_psi,
    kappa_phi,
    parallel_conductivity,
    lapse,
    d_lapse,
    d_shift,
    d_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
    sqrt_det_spatial_metric,
    extrinsic_curvature,
):
    tilde_j = tilde_j_drift(
        tilde_q,
        tilde_e,
        tilde_b,
        parallel_conductivity,
        lapse,
        sqrt_det_spatial_metric,
        spatial_metric,
    )
    return (
        -tilde_j
        - np.einsum("a, ai", tilde_e, d_shift)
        + tilde_psi
        * (
            np.einsum("a, ia", d_lapse, inv_spatial_metric)
            - lapse
            * trace_spatial_Gamma_second_kind(
                inv_spatial_metric, d_spatial_metric
            )
        )
    )


def source_tilde_b(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    kappa_psi,
    kappa_phi,
    parallel_conductivity,
    lapse,
    d_lapse,
    d_shift,
    d_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
    sqrt_det_spatial_metric,
    extrinsic_curvature,
):
    return -np.einsum("a, ai", tilde_b, d_shift) + tilde_phi * (
        np.einsum("a, ia", d_lapse, inv_spatial_metric)
        - lapse
        * trace_spatial_Gamma_second_kind(inv_spatial_metric, d_spatial_metric)
    )


def source_tilde_phi(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    kappa_psi,
    kappa_phi,
    parallel_conductivity,
    lapse,
    d_lapse,
    d_shift,
    d_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
    sqrt_det_spatial_metric,
    extrinsic_curvature,
):
    return np.einsum("a, a", tilde_b, d_lapse) - lapse * tilde_phi * (
        np.einsum("ab, ab", inv_spatial_metric, extrinsic_curvature) + kappa_phi
    )


def source_tilde_psi(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    kappa_psi,
    kappa_phi,
    parallel_conductivity,
    lapse,
    d_lapse,
    d_shift,
    d_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
    sqrt_det_spatial_metric,
    extrinsic_curvature,
):
    return (
        np.einsum("a, a", tilde_e, d_lapse)
        + lapse * tilde_q
        - lapse
        * tilde_psi
        * (
            np.einsum("ab, ab", inv_spatial_metric, extrinsic_curvature)
            + kappa_psi
        )
    )
