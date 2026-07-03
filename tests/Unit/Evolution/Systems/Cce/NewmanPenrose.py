# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def newman_penrose_alpha(
    bondi_j, eth_j, ethbar_j, bondi_k, bondi_r, bondi_q, eth_beta, one_minus_y
):
    one_plus_k = 1.0 + bondi_k
    sqrt_one_plus_k = np.sqrt(one_plus_k)
    q_plus_two_eth_beta = bondi_q + 2.0 * eth_beta

    return (
        one_minus_y
        / (32.0 * bondi_r)
        * (
            1.0
            / sqrt_one_plus_k
            * (
                (np.conj(bondi_j) ** 2 * eth_j) / (bondi_k * one_plus_k)
                + 1.0
                / bondi_k
                * (
                    bondi_j * np.conj(eth_j)
                    + np.conj(bondi_j) * ethbar_j
                    - np.conj(ethbar_j)
                )
                + (
                    2.0 * np.conj(bondi_j) * q_plus_two_eth_beta
                    - 3.0 * np.conj(ethbar_j)
                )
            )
            - 2.0 * sqrt_one_plus_k * np.conj(q_plus_two_eth_beta)
        )
    )


def newman_penrose_beta(
    bondi_j, eth_j, ethbar_j, bondi_k, bondi_r, bondi_q, eth_beta, one_minus_y
):
    one_plus_k = 1.0 + bondi_k
    sqrt_one_plus_k = np.sqrt(one_plus_k)
    q_plus_two_eth_beta = bondi_q + 2.0 * eth_beta

    return (
        one_minus_y
        / (32.0 * bondi_r)
        * (
            1.0
            / sqrt_one_plus_k
            * (
                (
                    -(bondi_j**2) * np.conj(eth_j) / (bondi_k * one_plus_k)
                    + 1.0
                    / bondi_k
                    * (
                        -bondi_j * np.conj(ethbar_j)
                        - np.conj(bondi_j) * eth_j
                        + ethbar_j
                    )
                    + (
                        2.0 * bondi_j * np.conj(q_plus_two_eth_beta)
                        - 3.0 * ethbar_j
                    )
                )
            )
            - 2.0 * sqrt_one_plus_k * q_plus_two_eth_beta
        )
    )


def newman_penrose_gamma(
    bondi_j,
    dy_j,
    eth_j,
    ethbar_j,
    bondi_k,
    bondi_h,
    bondi_r,
    bondi_u,
    eth_u,
    ethbar_u,
    bondi_w,
    dy_w,
    exp_2_beta,
    one_minus_y,
):
    one_plus_k = 1.0 + bondi_k

    return (
        1.0
        / (np.sqrt(32.0) * exp_2_beta)
        * (
            1.0
            / (2.0 * one_plus_k)
            * (
                one_minus_y
                * (one_minus_y / (2.0 * bondi_r) + bondi_w)
                * (np.conj(bondi_j) * dy_j - bondi_j * np.conj(dy_j))
                + (
                    2.0 * np.conj(bondi_h) * bondi_j
                    - 2.0 * bondi_h * np.conj(bondi_j)
                    + bondi_u
                    * (bondi_j * np.conj(eth_j) - np.conj(bondi_j) * ethbar_j)
                    + np.conj(bondi_u)
                    * (bondi_j * np.conj(ethbar_j) - np.conj(bondi_j) * eth_j)
                )
            )
            + 2.0 * one_minus_y * dy_w
            + (
                2.0 * bondi_w
                + bondi_j * np.conj(eth_u)
                - np.conj(bondi_j) * eth_u
                + bondi_k * (ethbar_u - np.conj(ethbar_u))
            )
        )
    )


def newman_penrose_epsilon(
    bondi_j, dy_j, bondi_k, bondi_r, dy_beta, one_minus_y
):
    return (
        one_minus_y**2
        / (np.sqrt(8.0) * bondi_r)
        * (
            dy_beta
            + (bondi_j * np.conj(dy_j) - np.conj(bondi_j) * dy_j)
            / (8.0 * (1.0 + bondi_k))
        )
    )


def newman_penrose_tau(
    bondi_j, bondi_k, bondi_r, bondi_q, eth_beta, one_minus_y
):
    one_plus_k = 1.0 + bondi_k
    sqrt_one_plus_k = np.sqrt(one_plus_k)
    two_eth_beta_minus_q = 2.0 * eth_beta - bondi_q

    return (
        one_minus_y
        / (8.0 * bondi_r)
        * (
            sqrt_one_plus_k * two_eth_beta_minus_q
            - bondi_j * np.conj(two_eth_beta_minus_q) / sqrt_one_plus_k
        )
    )


def newman_penrose_sigma(bondi_j, dy_j, bondi_k, bondi_r, one_minus_y):
    one_plus_k = 1.0 + bondi_k

    return (
        one_minus_y**2
        / (np.sqrt(128.0) * bondi_k * bondi_r)
        * (bondi_j**2 * np.conj(dy_j) / one_plus_k - one_plus_k * dy_j)
    )


def newman_penrose_rho(bondi_r, one_minus_y):
    return -one_minus_y / (np.sqrt(8.0) * bondi_r)


def newman_penrose_pi(
    bondi_j, bondi_k, bondi_r, bondi_q, eth_beta, one_minus_y
):
    one_plus_k = 1.0 + bondi_k
    sqrt_one_plus_k = np.sqrt(one_plus_k)
    q_plus_two_eth_beta = bondi_q + 2.0 * eth_beta

    return (
        one_minus_y
        / (8.0 * bondi_r)
        * (
            np.conj(bondi_j) * q_plus_two_eth_beta / sqrt_one_plus_k
            - sqrt_one_plus_k * np.conj(q_plus_two_eth_beta)
        )
    )


def newman_penrose_nu(bondi_j, bondi_k, eth_w, exp_2_beta):
    one_plus_k = 1.0 + bondi_k
    sqrt_one_plus_k = np.sqrt(one_plus_k)

    return (
        1.0
        / (2.0 * exp_2_beta)
        * (
            np.conj(bondi_j) * eth_w / sqrt_one_plus_k
            - sqrt_one_plus_k * np.conj(eth_w)
        )
    )


def newman_penrose_mu(bondi_r, bondi_w, ethbar_u, exp_2_beta, one_minus_y):
    return (
        1.0
        / (np.sqrt(8.0) * exp_2_beta)
        * (np.conj(ethbar_u) + ethbar_u - one_minus_y / bondi_r - 2.0 * bondi_w)
    )


def newman_penrose_lambda(
    bondi_j,
    dy_j,
    eth_j,
    ethbar_j,
    bondi_k,
    bondi_h,
    bondi_r,
    bondi_u,
    eth_u,
    ethbar_u,
    bondi_w,
    exp_2_beta,
    one_minus_y,
):
    one_plus_k = 1.0 + bondi_k

    inner1 = (
        one_minus_y
        / (2.0 * one_plus_k)
        * (
            (np.conj(bondi_j) ** 2 * dy_j - np.conj(dy_j)) / bondi_k
            - (2.0 + bondi_k) * np.conj(dy_j)
        )
    )

    inner2 = 2.0 * bondi_h + bondi_u * ethbar_j + np.conj(bondi_u) * eth_j

    return (
        1.0
        / (np.sqrt(32.0) * exp_2_beta)
        * (
            (one_minus_y / bondi_r + 2.0 * bondi_w) * inner1
            + 2.0 * one_plus_k * np.conj(eth_u)
            + (
                np.conj(inner2)
                + 2.0 * np.conj(bondi_j) * (ethbar_u - np.conj(ethbar_u))
            )
            + np.conj(inner2) / bondi_k
            - np.conj(bondi_j) ** 2
            * (inner2 + 2.0 * bondi_k * eth_u)
            / (bondi_k * one_plus_k)
        )
    )


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


def psi2(
    bondi_j,
    bondi_k,
    bondi_r,
    dy_mu,
    eth_pi,
    ethbar_pi,
    np_alpha,
    np_beta,
    np_epsilon,
    np_sigma,
    np_rho,
    np_pi,
    np_mu,
    np_lambda,
    one_minus_y,
):
    sqrt_one_plus_k = np.sqrt(1.0 + bondi_k)

    return (
        0.25
        * one_minus_y
        / bondi_r
        * (
            np.sqrt(2.0) * one_minus_y * dy_mu
            + sqrt_one_plus_k * eth_pi
            - bondi_j * ethbar_pi / sqrt_one_plus_k
        )
        + (np_epsilon + np.conj(np_epsilon) - np.conj(np_rho)) * np_mu
        + (np.conj(np_alpha) - np_beta - np.conj(np_pi)) * np_pi
        - np_sigma * np_lambda
    )


def newman_penrose_d_psi1(bondi_r, dy_psi_1, one_minus_y):
    return dy_psi_1 * one_minus_y**2 / (2 * np.sqrt(2) * bondi_r)


def newman_penrose_deltabar_psi0(
    bondi_j, bondi_k, bondi_r, eth_psi_0, ethbar_psi_0, one_minus_y
):
    sqrt_one_plus_k = np.sqrt(1.0 + bondi_k)

    return (
        -one_minus_y
        * (
            sqrt_one_plus_k * ethbar_psi_0 / np.sqrt(2)
            - np.conj(bondi_j) * eth_psi_0 / (np.sqrt(2) * sqrt_one_plus_k)
        )
        / (2 * np.sqrt(2) * bondi_r)
    )


def bianchi_constraint_d_psi1(
    np_alpha,
    np_epsilon,
    np_rho,
    np_pi,
    psi_0,
    psi_1,
    np_d_psi_1,
    np_deltabar_psi_0,
):
    return (
        np_d_psi_1
        - np_deltabar_psi_0
        + (4 * np_alpha - np_pi) * psi_0
        - 2 * (2 * np_rho + np_epsilon) * psi_1
    )


def newman_penrose_d_psi2(bondi_r, dy_psi_2, one_minus_y):
    return dy_psi_2 * one_minus_y**2 / (2 * np.sqrt(2) * bondi_r)


def newman_penrose_deltabar_psi1(
    bondi_j, bondi_k, bondi_r, eth_psi_1, ethbar_psi_1, one_minus_y
):
    sqrt_one_plus_k = np.sqrt(1.0 + bondi_k)

    return (
        -one_minus_y
        * (
            sqrt_one_plus_k * ethbar_psi_1 / np.sqrt(2)
            - np.conj(bondi_j) * eth_psi_1 / (np.sqrt(2) * sqrt_one_plus_k)
        )
        / (2 * np.sqrt(2) * bondi_r)
    )


def bianchi_constraint_d_psi2(
    np_alpha,
    np_lambda,
    np_rho,
    np_pi,
    psi_0,
    psi_1,
    psi_2,
    np_d_psi_2,
    np_deltabar_psi_1,
):
    return (
        np_d_psi_2
        + np_lambda * psi_0
        - np_deltabar_psi_1
        - 2 * (np_pi - np_alpha) * psi_1
        - 3 * np_rho * psi_2
    )
