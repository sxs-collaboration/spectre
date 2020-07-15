# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def compute_News(linear_coefficient, quadratic_coefficient, time,
                 news_coefficient, _1, _2, _3, _4, _5, _6):
    return news_coefficient * (1.0 + linear_coefficient * time +
                               quadratic_coefficient * time**2)


def compute_EthInertialRetardedTime(linear_coefficient, quadratic_coefficient,
                                    time, _1, _2, _3, _4, _5, _6,
                                    eth_u_coefficient):
    return eth_u_coefficient * (1.0 + linear_coefficient * time +
                                quadratic_coefficient * time**2)


def compute_Du_TimeIntegral_ScriPlus_Psi4(linear_coefficient,
                                          quadratic_coefficient, time, _1, _2,
                                          _3, _4, _5, psi4_coefficient, _6):
    return psi4_coefficient * (linear_coefficient +
                               2.0 * quadratic_coefficient * time)


def compute_ScriPlus_Psi3(linear_coefficient, quadratic_coefficient, time, _1,
                          psi3_coefficient, _2, _3, _4, psi4_coefficient,
                          eth_u_coefficient):
    time_factor = (1.0 + linear_coefficient * time +
                   quadratic_coefficient * time**2)
    psi4 = psi4_coefficient * (linear_coefficient +
                               2.0 * quadratic_coefficient * time)
    psi3 = psi3_coefficient * time_factor
    eth_u = eth_u_coefficient * time_factor
    return psi3 + 0.5 * eth_u * psi4


def compute_ScriPlus_Psi2(linear_coefficient, quadratic_coefficient, time, _1,
                          psi3_coefficient, psi2_coefficient, _2, _3,
                          psi4_coefficient, eth_u_coefficient):
    time_factor = (1.0 + linear_coefficient * time +
                   quadratic_coefficient * time**2)
    psi4 = psi4_coefficient * (linear_coefficient +
                               2.0 * quadratic_coefficient * time)
    psi3 = psi3_coefficient * time_factor
    psi2 = psi2_coefficient * time_factor
    eth_u = eth_u_coefficient * time_factor
    return psi2 + psi3 * eth_u + 0.25 * psi4 * eth_u**2


def compute_ScriPlus_Psi1(linear_coefficient, quadratic_coefficient, time, _1,
                          psi3_coefficient, psi2_coefficient, psi1_coefficient,
                          _2, psi4_coefficient, eth_u_coefficient):
    time_factor = (1.0 + linear_coefficient * time +
                   quadratic_coefficient * time**2)
    psi4 = psi4_coefficient * (linear_coefficient +
                               2.0 * quadratic_coefficient * time)
    psi3 = psi3_coefficient * time_factor
    psi2 = psi2_coefficient * time_factor
    psi1 = psi1_coefficient * time_factor
    eth_u = eth_u_coefficient * time_factor
    return psi1 + 1.5 * psi2 * eth_u + 0.75 * psi3 * eth_u**2 \
        + 0.125 * psi4 * eth_u**3


def compute_ScriPlus_Psi0(linear_coefficient, quadratic_coefficient, time, _1,
                          psi3_coefficient, psi2_coefficient, psi1_coefficient,
                          psi0_coefficient, psi4_coefficient,
                          eth_u_coefficient):
    time_factor = (1.0 + linear_coefficient * time +
                   quadratic_coefficient * time**2)
    psi4 = psi4_coefficient * (linear_coefficient +
                               2.0 * quadratic_coefficient * time)
    psi3 = psi3_coefficient * time_factor
    psi2 = psi2_coefficient * time_factor
    psi1 = psi1_coefficient * time_factor
    psi0 = psi0_coefficient * time_factor
    eth_u = eth_u_coefficient * time_factor
    return psi0 + 2.0 * psi1 * eth_u + 0.75 * psi2 * eth_u**2 \
        + 0.5 * psi3 * eth_u**3 + 0.0625 * psi4 * eth_u**4
