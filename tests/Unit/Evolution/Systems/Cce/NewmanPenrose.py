# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def psi0(bondi_j, dy_j, dy_dy_j, bondi_k, bondi_r, one_minus_y):
    dy_beta = 0.125 * one_minus_y * (
        dy_j * np.conj(dy_j) - 0.25 *
        (bondi_j * np.conj(dy_j) + np.conj(bondi_j) * dy_j)**2 / bondi_k**2)
    return one_minus_y**4 * 1.0 / (16.0 * bondi_r**2) * (
        (1.0 + bondi_k) * dy_beta * dy_j / bondi_k -
        bondi_j**2 * dy_beta * np.conj(dy_j) /
        (bondi_k + bondi_k**2) - bondi_j * np.conj(bondi_j)**2 * dy_j**2 /
        (4.0 * bondi_k**3) - bondi_j**3 * np.conj(dy_j)**2 /
        (4.0 * bondi_k**3) + 0.5 * (-1.0 - 1.0 / bondi_k) * dy_dy_j +
        0.5 * bondi_j**2 * np.conj(dy_dy_j) /
        (bondi_k**2 + bondi_k) + 0.5 * bondi_j *
        (1.0 + bondi_k**2) * dy_j * np.conj(dy_j) / bondi_k**3)
