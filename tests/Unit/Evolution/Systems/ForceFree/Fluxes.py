# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def levi_civita_symbol(i, j, k):
    return np.sign(j - i) * np.sign(k - i) * np.sign(k - j)


def tilde_e_flux(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    tilde_j,
    lapse,
    shift,
    sqrt_det_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
):
    magnetic_field_one_form = (
        np.einsum("a, ia", tilde_b, spatial_metric) / sqrt_det_spatial_metric
    )

    result = -np.outer(shift, tilde_e) + lapse * (
        inv_spatial_metric * tilde_psi
    )

    for j in range(3):
        for i in range(3):
            for k in range(3):
                result[j, i] -= (
                    lapse
                    * levi_civita_symbol(i, j, k)
                    * magnetic_field_one_form[k]
                )

    return result


def tilde_b_flux(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    tilde_j,
    lapse,
    shift,
    sqrt_det_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
):
    electric_field_one_form = (
        np.einsum("a, ia", tilde_e, spatial_metric) / sqrt_det_spatial_metric
    )

    result = -np.outer(shift, tilde_b) + lapse * (
        inv_spatial_metric * tilde_phi
    )

    for j in range(3):
        for i in range(3):
            for k in range(3):
                result[j, i] += (
                    lapse
                    * levi_civita_symbol(i, j, k)
                    * electric_field_one_form[k]
                )

    return result


def tilde_psi_flux(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    tilde_j,
    lapse,
    shift,
    sqrt_det_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
):
    return -shift * tilde_psi + lapse * tilde_e


def tilde_phi_flux(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    tilde_j,
    lapse,
    shift,
    sqrt_det_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
):
    return -shift * tilde_phi + lapse * tilde_b


def tilde_q_flux(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    tilde_j,
    lapse,
    shift,
    sqrt_det_spatial_metric,
    spatial_metric,
    inv_spatial_metric,
):
    return tilde_j - shift * tilde_q
