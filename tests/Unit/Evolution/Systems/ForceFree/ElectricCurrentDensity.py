# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def levi_civita_symbol(i, j, k):
    return np.sign(j - i) * np.sign(k - i) * np.sign(k - j)


def tilde_j_drift(
    tilde_q,
    tilde_e,
    tilde_b,
    parallel_conductivity,
    lapse,
    sqrt_det_spatial_metric,
    spatial_metric,
):
    tilde_e_one_form = np.einsum("a, ia", tilde_e, spatial_metric)
    tilde_b_one_form = np.einsum("a, ia", tilde_b, spatial_metric)
    tilde_b_squared = np.einsum("a, a", tilde_b_one_form, tilde_b)

    result = tilde_e * 0.0

    for i in range(3):
        for j in range(3):
            for k in range(3):
                e_ijk = levi_civita_symbol(i, j, k) / sqrt_det_spatial_metric
                result[i] += e_ijk * tilde_e_one_form[j] * tilde_b_one_form[k]

    # overall scaling
    result = result * lapse * tilde_q / tilde_b_squared

    return result


def tilde_j_parallel(
    tilde_q,
    tilde_e,
    tilde_b,
    parallel_conductivity,
    lapse,
    sqrt_det_spatial_metric,
    spatial_metric,
):
    tilde_e_one_form = np.einsum("a, ia", tilde_e, spatial_metric)
    tilde_b_one_form = np.einsum("a, ia", tilde_b, spatial_metric)

    tilde_e_squared = np.einsum("a, a", tilde_e_one_form, tilde_e)
    tilde_b_squared = np.einsum("a, a", tilde_b_one_form, tilde_b)
    tilde_e_dot_tilde_b = np.einsum("a, a", tilde_e_one_form, tilde_b)

    return (
        parallel_conductivity
        * lapse
        * (
            tilde_e_dot_tilde_b * tilde_b
            + max(tilde_e_squared - tilde_b_squared, 0.0) * tilde_e
        )
        / tilde_b_squared
    )


def tilde_j(
    tilde_q,
    tilde_e,
    tilde_b,
    parallel_conductivity,
    lapse,
    sqrt_det_spatial_metric,
    spatial_metric,
):
    return tilde_j_drift(
        tilde_q,
        tilde_e,
        tilde_b,
        parallel_conductivity,
        lapse,
        sqrt_det_spatial_metric,
        spatial_metric,
    ) + tilde_j_parallel(
        tilde_q,
        tilde_e,
        tilde_b,
        parallel_conductivity,
        lapse,
        sqrt_det_spatial_metric,
        spatial_metric,
    )
