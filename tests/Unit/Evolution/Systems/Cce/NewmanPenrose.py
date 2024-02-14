# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def psi0(bondi_j, dy_j, dy_dy_j, bondi_k, bondi_r, one_minus_y):
    dy_beta = (
        0.125
        * one_minus_y
        * (
            dy_j * np.conj(dy_j)
            - 0.25
            * (bondi_j * np.conj(dy_j) + np.conj(bondi_j) * dy_j) ** 2
            / bondi_k**2
        )
    )
    return (
        one_minus_y**4
        * 1.0
        / (16.0 * bondi_r**2)
        * (
            (1.0 + bondi_k) * dy_beta * dy_j / bondi_k
            - bondi_j**2 * dy_beta * np.conj(dy_j) / (bondi_k + bondi_k**2)
            - bondi_j * np.conj(bondi_j) ** 2 * dy_j**2 / (4.0 * bondi_k**3)
            - bondi_j**3 * np.conj(dy_j) ** 2 / (4.0 * bondi_k**3)
            + 0.5 * (-1.0 - 1.0 / bondi_k) * dy_dy_j
            + 0.5 * bondi_j**2 * np.conj(dy_dy_j) / (bondi_k**2 + bondi_k)
            + 0.5
            * bondi_j
            * (1.0 + bondi_k**2)
            * dy_j
            * np.conj(dy_j)
            / bondi_k**3
        )
    )


def psi1(
    bondi_j,
    dy_j,
    bondi_k,
    bondi_q,
    dy_q,
    bondi_r,
    eth_r_divided_by_r,
    dy_beta,
    eth_beta,
    eth_dy_beta,
    one_minus_y,
):
    prefac = 1.0 / np.sqrt(128.0)
    one_plus_k = 1.0 + bondi_k
    eth_beta_plus_half_q = eth_beta + 0.5 * bondi_q
    conj_j_times_dy_j = np.conj(bondi_j) * dy_j

    inner_expr = bondi_j * (
        -2.0 * np.conj(dy_q)
        + np.conj(dy_j)
        * (2.0 * eth_beta_plus_half_q + bondi_j * np.conj(eth_beta_plus_half_q))
    ) + one_plus_k * (
        eth_beta_plus_half_q * (conj_j_times_dy_j - np.conj(conj_j_times_dy_j))
        + 2.0 * (dy_q + bondi_j * np.conj(dy_q))
        - one_plus_k * (2.0 * dy_q + dy_j * np.conj(eth_beta_plus_half_q))
    )

    return (
        prefac
        * one_minus_y**2
        / (bondi_r**2 * np.sqrt(one_plus_k))
        * (
            bondi_j * np.conj(eth_beta_plus_half_q)
            - one_plus_k * eth_beta_plus_half_q
            + one_minus_y
            * (
                eth_dy_beta * one_plus_k
                - bondi_j * np.conj(eth_dy_beta)
                + dy_beta
                * (
                    one_plus_k * eth_r_divided_by_r
                    - bondi_j * np.conj(eth_r_divided_by_r)
                )
                + 0.25 * inner_expr / bondi_k
            )
        )
    )
