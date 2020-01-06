# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Test function for beta equation


def integrand_for_beta(dy_j, j, one_minus_y):
    dy_jbar = np.conj(dy_j)
    jbar = np.conj(j)
    dy_j_jbar = j * dy_jbar + jbar * dy_j
    k_squared = 1.0 + j * jbar
    return one_minus_y / 8.0 * (dy_j * dy_jbar \
                                - dy_j_jbar**2 / (4.0 * k_squared))


# Test functions for Q equation


def integrand_for_q_pole_part(eth_beta):
    return -4.0 * eth_beta


def integrand_for_q_regular_part(_, dy_beta, dy_j, j, eth_dy_beta, eth_j_jbar,
                                 eth_jbar_dy_j, ethbar_dy_j, ethbar_j,
                                 eth_r_divided_by_r, k):
    # script_aq input is unused, needed only to match function signature from
    # the C++
    eth_dy_jbar = np.conj(ethbar_dy_j)
    dy_jbar = np.conj(dy_j)
    ethbar_r_divided_by_r = np.conj(eth_r_divided_by_r)
    eth_jbar = np.conj(ethbar_j)
    jbar = np.conj(j)
    script_aq = - eth_jbar_dy_j / 4.0 + j * eth_dy_jbar / 4.0 \
                - eth_jbar * dy_j / 4.0 \
                + eth_j_jbar * (jbar * dy_j + j * dy_jbar)\
                  / (8.0 * (1.0 + j * jbar)) \
                + (j * dy_jbar - jbar * dy_j) * eth_r_divided_by_r / 4.0
    return - (2.0 * script_aq + 2.0 * j * np.conj(script_aq) / k \
              - 2.0 * eth_dy_beta + ethbar_dy_j / k \
              - 2.0 * dy_beta * eth_r_divided_by_r \
              + dy_j * ethbar_r_divided_by_r / k)


# Test function for U equation


def integrand_for_u(exp_2_beta, j, q, k, r):
    qbar = np.conj(q)
    return exp_2_beta / (2.0 * r) * (k * q - j * qbar)


# Test functions for W equation


def integrand_for_w_pole_part(ethbar_u):
    eth_ubar = np.conj(ethbar_u)
    return eth_ubar + ethbar_u


def integrand_for_w_regular_part(_, dy_u, exp_2_beta, j, q, eth_beta,
                                 eth_eth_beta, eth_ethbar_beta, eth_ethbar_j,
                                 eth_ethbar_j_jbar, eth_j_jbar, ethbar_dy_u,
                                 ethbar_ethbar_j, ethbar_j, eth_r_divided_by_r,
                                 k, r):
    # script_av input is unused, needed only to match function signature from
    # the C++
    dy_ubar = np.conj(dy_u)
    ethbar_beta = np.conj(eth_beta)
    eth_dy_ubar = np.conj(ethbar_dy_u)
    ethbar_ethbar_beta = np.conj(eth_eth_beta)
    eth_eth_jbar = np.conj(ethbar_ethbar_j)
    eth_jbar = np.conj(ethbar_j)
    ethbar_j_jbar = np.conj(eth_j_jbar)
    ethbar_r_divided_by_r = np.conj(eth_r_divided_by_r)
    jbar = np.conj(j)
    k_squared = (1.0 + j * jbar)
    qbar = np.conj(q)
    script_av = eth_beta * eth_jbar + ethbar_ethbar_j / 2.0 \
                + j * ethbar_beta**2 + j * ethbar_ethbar_beta \
                + eth_j_jbar * ethbar_j_jbar / (8.0 * k_squared * k) \
                + 1 / (2.0 * k) - eth_ethbar_j_jbar / (8.0 * k) \
                - eth_j_jbar * ethbar_beta / (2.0 * k) \
                - eth_jbar * ethbar_j / (4.0 * k) \
                - eth_ethbar_j * jbar / (4.0 * k) + k / 2.0 \
                - eth_ethbar_beta * k - eth_beta * ethbar_beta * k  \
                + 1.0 / 4.0 * (- k * q * qbar + j * qbar**2)
    return 1.0 / 4.0 * eth_dy_ubar + 1.0 / 4.0 * ethbar_dy_u  \
           + 1.0 / 4.0 * dy_u * ethbar_r_divided_by_r \
           + 1.0 / 4.0 * dy_ubar * eth_r_divided_by_r - 1.0 / (2.0 * r) \
           + exp_2_beta * (script_av + np.conj(script_av)) / (4.0 * r)


# Test functions for H equation


def integrand_for_h_pole_part(j, u, w, eth_u, ethbar_j, ethbar_jbar_u,
                              ethbar_u, k):
    eth_ubar = np.conj(ethbar_u)
    eth_j_ubar = np.conj(ethbar_jbar_u)
    return - 1.0 / 2.0 * eth_j_ubar - j * eth_ubar - 1.0 / 2.0 * j * ethbar_u \
           - k * eth_u - 1.0 / 2.0 * u * ethbar_j + 2.0 * j * w


def integrand_for_h_regular_part(
    _0, _1, _2, dy_dy_j, dy_j, dy_w, exp_2_beta, j, q, u, w, eth_beta,
    eth_eth_beta, eth_ethbar_beta, eth_ethbar_j, eth_ethbar_j_jbar, eth_j_jbar,
    eth_q, eth_u, eth_ubar_dy_j, ethbar_dy_j, ethbar_ethbar_j, ethbar_j,
    ethbar_jbar_dy_j, ethbar_jbar_q_minus_2_eth_beta, ethbar_q, ethbar_u,
    du_r_divided_by_r, eth_r_divided_by_r, k, one_minus_y, r):
    # script_aj, script_bj, and script_cj input is unused, needed only to match
    # function signature from the C++
    jbar = np.conj(j)
    k_squared = 1.0 + j * jbar
    eth_eth_jbar = np.conj(ethbar_ethbar_j)
    ethbar_eth_jbar = np.conj(eth_ethbar_j)
    eth_jbar = np.conj(ethbar_j)
    ethbar_j_jbar = np.conj(eth_j_jbar)

    script_aj = 1.0 / 4.0 * eth_eth_jbar - 1.0 / (4.0 * k * k_squared) \
                - eth_ethbar_j_jbar / (16.0 * k * k_squared) \
                + j * ethbar_eth_jbar / (16.0 * k * k_squared) \
                + 3.0 / (4.0 * k) - eth_ethbar_beta / (4.0 * k) \
                - eth_ethbar_j * jbar * (1.0 - 1.0 / (4.0 * k_squared)) \
                  / (4.0 * k) \
                + 1.0 / 2.0 * eth_jbar * (eth_beta + j * ethbar_j_jbar \
                                 / (4.0 * k * k_squared) \
                               - ethbar_j * (-1.0 + 2.0 * k_squared) \
                                 / (4.0 * k * k_squared) - 1.0 / 2.0 * q)

    dy_jbar = np.conj(dy_j)
    dy_j_jbar = j * dy_jbar + jbar * dy_j
    eth_dy_jbar = np.conj(ethbar_dy_j)
    eth_j_dy_jbar = np.conj(ethbar_jbar_dy_j)
    ubar = np.conj(u)

    script_bj = (- eth_u * jbar * dy_j_jbar / (4.0 * k) + 1.0 / 2.0 * dy_w \
                 + 1.0 / (4.0 * r) + 1.0 / 4.0 * ethbar_j * dy_jbar * u \
                 - ethbar_j_jbar * dy_j_jbar * u / (8.0 * k_squared) \
                 - 1.0 / 4.0 * j * eth_dy_jbar * ubar \
                 + 1.0 / 4.0 * eth_j_dy_jbar * ubar) \
                + one_minus_y \
                  * (du_r_divided_by_r * dy_j * 1.0 / 4.0 \
                       * (-2.0 * dy_jbar + jbar * dy_j_jbar / k_squared) \
                     - 1.0/4.0 * dy_j * dy_jbar * w \
                     + w * dy_j_jbar**2 / (16.0 * k_squared)) \
                + one_minus_y**2 * (- dy_j * dy_jbar / (8.0 * r) \
                                    + dy_j_jbar**2 / (32.0 * k_squared * r))

    script_cj = 1.0 / 2.0 * ethbar_j * k * (eth_beta - 1.0 / 2.0 * q)

    eth_j_qbar_minus_2_ethbar_beta = np.conj(ethbar_jbar_q_minus_2_eth_beta)
    eth_qbar = np.conj(ethbar_q)
    ethbar_r_divided_by_r = np.conj(eth_r_divided_by_r)
    ethbar_ubar = np.conj(eth_u)
    eth_ubar = np.conj(ethbar_u)
    ubar = np.conj(u)
    dy_jbar = np.conj(dy_j)

    return -1.0 / 2.0 * eth_ubar_dy_j - 1.0 / 2.0 * u * ethbar_dy_j \
           - 1.0 / 2.0 * u * dy_j * ethbar_r_divided_by_r \
           + j * (script_bj + np.conj(script_bj)) \
           + exp_2_beta / (2.0 * r) \
             * (script_cj + j**2 / k_squared * np.conj(script_cj) \
                + eth_eth_beta - 1.0 / 2.0 * eth_q \
                - j * (script_aj + np.conj(script_aj)) \
                + eth_j_qbar_minus_2_ethbar_beta / (4.0 * k) \
                - j * eth_qbar / (4.0 * k) + (eth_beta - 1.0 / 2.0 * q)**2) \
           - dy_j \
             * (jbar * eth_u / (2.0 * k) - 1.0 / 2.0 * ethbar_ubar * j * k \
                + 1.0 / 4.0 * (eth_ubar - ethbar_u) * k_squared + \
                + 1.0 / 2.0 * eth_r_divided_by_r * ubar - 1.0 / 2.0 * w) \
           + dy_jbar \
             * (-1.0 / 4.0 * j**2 * (- eth_ubar + ethbar_u) \
                + eth_u * j * (j * jbar /( 2.0 * k))) \
           + one_minus_y \
             * (1.0 / 2.0 * (-dy_j / r + 2.0 * du_r_divided_by_r * dy_dy_j \
                             + w * dy_dy_j) + dy_j * (1.0 / 2.0 * dy_w \
                                                      + 1.0 / (2.0 * r))) \
           + one_minus_y**2 * dy_dy_j / (4.0 * r)


def linear_factor_for_h(_, dy_j, j, one_minus_y):
    # script_dj input is unused, needed only to match function signature from
    # the C++
    dy_jbar = np.conj(dy_j)
    jbar = np.conj(j)
    k_squared = 1.0 + j * np.conj(j)

    return 1.0 + ((1.0 / 4.0) * np.conj(one_minus_y) * j \
                  * (-2.0 * dy_jbar + jbar * (jbar * dy_j
                                              + j * dy_jbar) / k_squared))


def linear_factor_for_conjugate_h(_, dy_j, j, one_minus_y):
    # script_dj input is unused, needed only to match function signature from
    # the C++
    dy_jbar = np.conj(dy_j)
    jbar = np.conj(j)
    k_squared = 1.0 + j * np.conj(j)

    return (1.0 / 4.0) * one_minus_y * j \
        * np.conj(-2.0 * dy_jbar + jbar \
                  * (jbar * dy_j + j * dy_jbar) / k_squared)
