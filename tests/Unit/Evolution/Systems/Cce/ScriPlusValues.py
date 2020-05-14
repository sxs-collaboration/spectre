# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def news(dy_du_bondi_j, beta, eth_beta, eth_eth_beta, boundary_r):
    return 2.0 * np.conj(-boundary_r * np.exp(-2.0 * beta) * dy_du_bondi_j +
                         eth_eth_beta + 2.0 * eth_beta**2)


def time_integral_psi_4(exp_2_beta, dy_bondi_u, eth_dy_bondi_u, dy_du_bondi_j,
                        boundary_r, eth_r_divided_by_r):
    return (boundary_r / exp_2_beta) * np.conj(
        eth_dy_bondi_u + eth_r_divided_by_r * dy_bondi_u + dy_du_bondi_j)


def constant_factor_psi_4(exp_2_beta):
    return 2.0 / exp_2_beta


def psi_3(exp_2_beta, eth_beta, eth_ethbar_beta, ethbar_eth_ethbar_beta,
          dy_du_bondi_j, ethbar_dy_du_bondi_j, boundary_r, eth_r_divided_by_r):
    return 2.0 * np.conj(eth_beta) + 4.0 * np.conj(
        eth_beta
    ) * eth_ethbar_beta + ethbar_eth_ethbar_beta + 2.0 * boundary_r * (
        np.conj(dy_du_bondi_j) * eth_beta / exp_2_beta) - 2.0 * boundary_r * (
            0.5 / exp_2_beta) * (np.conj(ethbar_dy_du_bondi_j) +
                                 eth_r_divided_by_r * np.conj(dy_du_bondi_j))


def psi_2(exp_2_beta, dy_bondi_q, ethbar_dy_bondi_q, dy_bondi_u,
          eth_dy_bondi_u, dy_dy_bondi_u, ethbar_dy_dy_bondi_u, dy_dy_bondi_w,
          dy_bondi_j, dy_du_bondi_j, boundary_r, eth_r_divided_by_r):
    return -0.25 / exp_2_beta * (
        exp_2_beta * (-2.0 * boundary_r) *
        (np.conj(ethbar_dy_bondi_q) + eth_r_divided_by_r * np.conj(dy_bondi_q))
        + 2.0 * boundary_r**2 *
        (np.conj(ethbar_dy_dy_bondi_u) + 2.0 * eth_r_divided_by_r *
         np.conj(dy_dy_bondi_u) + ethbar_dy_dy_bondi_u +
         2.0 * np.conj(eth_r_divided_by_r) * dy_dy_bondi_u) +
        (-2.0 * boundary_r * dy_bondi_j) *
        (-2.0 * boundary_r *
         (np.conj(eth_dy_bondi_u) + np.conj(eth_r_divided_by_r * dy_bondi_u) +
          np.conj(dy_du_bondi_j))) - 4.0 * boundary_r**2 * dy_dy_bondi_w)


def psi_1(dy_dy_bondi_beta, eth_dy_dy_bondi_beta, dy_bondi_j, dy_bondi_q,
          dy_dy_bondi_q, boundary_r, eth_r_divided_by_r):
    return 0.125 * (
        -12.0 * (2.0 * boundary_r**2) *
        (eth_dy_dy_bondi_beta + 2.0 * eth_r_divided_by_r * dy_dy_bondi_beta) +
        (-2.0 * boundary_r * dy_bondi_j) *
        (-2.0 * boundary_r * np.conj(dy_bondi_q)) + 2.0 *
        (2.0 * boundary_r**2 * dy_dy_bondi_q))


def psi_0(dy_bondi_j, dy_dy_dy_bondi_j, boundary_r):
    return 1.5 * (0.25 * (-2.0 * boundary_r * np.conj(dy_bondi_j)) *
                  (-2.0 * boundary_r * dy_bondi_j)**2 +
                  4.0 / 3.0 * boundary_r**3 * dy_dy_dy_bondi_j)


def strain(dy_bondi_j, eth_eth_retarded_time, boundary_r):
    return -2.0 * np.conj(
        boundary_r * dy_bondi_j) + np.conj(eth_eth_retarded_time)
