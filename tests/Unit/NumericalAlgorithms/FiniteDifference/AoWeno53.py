# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import Reconstruction


def test_aoweno53(u, extents, dim):
    gamma_hi = 0.85
    gamma_lo = 0.999
    epsilon = 1.0e-12
    exponent = 8

    def aoweno53(q):
        j = 2
        moments_sr3_1 = [
            1.041666666666666 * q[j] - 0.08333333333333333 * q[j - 1] +
            0.04166666666666666 * q[j - 2],
            0.5 * q[j - 2] - 2.0 * q[j - 1] + 1.5 * q[j],
            0.5 * q[j - 2] - q[j - 1] + 0.5 * q[j]
        ]
        moments_sr3_2 = [
            0.04166666666666666 * q[j + 1] + 0.9166666666666666 * q[j] +
            0.04166666666666666 * q[j - 1], 0.5 * (q[j + 1] - q[j - 1]),
            0.5 * q[j - 1] - q[j] + 0.5 * q[j + 1]
        ]
        moments_sr3_3 = [
            0.04166666666666666 * q[j + 2] - 0.08333333333333333 * q[j + 1] +
            1.04166666666666666 * q[j],
            -1.5 * q[j] + 2.0 * q[j + 1] - 0.5 * q[j + 2],
            0.5 * q[j] - q[j + 1] + 0.5 * q[j + 2]
        ]
        moments_sr5 = [
            -2.95138888888888881e-03 * q[j - 2] +
            5.34722222222222196e-02 * q[j - 1] +
            8.98958333333333304e-01 * q[j] +
            5.34722222222222196e-02 * q[j + 1] +
            -2.95138888888888881e-03 * q[j + 2],
            7.08333333333333315e-02 * q[j - 2] +
            -6.41666666666666718e-01 * q[j - 1] +
            6.41666666666666718e-01 * q[j + 1] +
            -7.08333333333333315e-02 * q[j + 2],
            -3.27380952380952397e-02 * q[j - 2] +
            6.30952380952380931e-01 * q[j - 1] +
            -1.19642857142857140e+00 * q[j] +
            6.30952380952380931e-01 * q[j + 1] +
            -3.27380952380952397e-02 * q[j + 2],
            -8.33333333333333287e-02 * q[j - 2] +
            1.66666666666666657e-01 * q[j - 1] +
            -1.66666666666666657e-01 * q[j + 1] +
            8.33333333333333287e-02 * q[j + 2],
            4.16666666666666644e-02 * q[j - 2] +
            -1.66666666666666657e-01 * q[j - 1] +
            2.50000000000000000e-01 * q[j] +
            -1.66666666666666657e-01 * q[j + 1] +
            4.16666666666666644e-02 * q[j + 2]
        ]

        beta_r3_1 = moments_sr3_1[1]**2 + 37.0 / 3.0 * moments_sr3_1[2]**2
        beta_r3_2 = moments_sr3_2[1]**2 + 37.0 / 3.0 * moments_sr3_2[2]**2
        beta_r3_3 = moments_sr3_3[1]**2 + 37.0 / 3.0 * moments_sr3_3[2]**2
        beta_sr5 = (moments_sr5[1]**2 +
                    61.0 / 5.0 * moments_sr5[1] * moments_sr5[3] +
                    37.0 / 3.0 * moments_sr5[2]**2 +
                    1538.0 / 7.0 * moments_sr5[2] * moments_sr5[4] +
                    8973.0 / 50.0 * moments_sr5[3]**2 +
                    167158.0 / 49.0 * moments_sr5[4]**2)

        linear_weights = [
            gamma_hi, 0.5 * (1.0 - gamma_hi) * (1.0 - gamma_lo),
            (1.0 - gamma_hi) * gamma_lo,
            0.5 * (1.0 - gamma_hi) * (1.0 - gamma_lo)
        ]
        nonlinear_weights = np.asarray([
            linear_weights[0] / (beta_sr5 + epsilon)**exponent,
            linear_weights[1] / (beta_r3_1 + epsilon)**exponent,
            linear_weights[2] / (beta_r3_2 + epsilon)**exponent,
            linear_weights[3] / (beta_r3_3 + epsilon)**exponent
        ])
        normalization = np.sum(nonlinear_weights)
        nonlinear_weights = nonlinear_weights / normalization

        moments = np.asarray([
            nonlinear_weights[0] / linear_weights[0] *
            (moments_sr5[0] - linear_weights[1] * moments_sr3_1[0] -
             linear_weights[2] * moments_sr3_2[0] -
             linear_weights[3] * moments_sr3_3[0]) +
            nonlinear_weights[1] * moments_sr3_1[0] +
            nonlinear_weights[2] * moments_sr3_2[0] +
            nonlinear_weights[3] * moments_sr3_3[0],
            nonlinear_weights[0] / linear_weights[0] *
            (moments_sr5[1] - linear_weights[1] * moments_sr3_1[1] -
             linear_weights[2] * moments_sr3_2[1] -
             linear_weights[3] * moments_sr3_3[1]) +
            nonlinear_weights[1] * moments_sr3_1[1] +
            nonlinear_weights[2] * moments_sr3_2[1] +
            nonlinear_weights[3] * moments_sr3_3[1],
            nonlinear_weights[0] / linear_weights[0] *
            (moments_sr5[2] - linear_weights[1] * moments_sr3_1[2] -
             linear_weights[2] * moments_sr3_2[2] -
             linear_weights[3] * moments_sr3_3[2]) +
            nonlinear_weights[1] * moments_sr3_1[2] +
            nonlinear_weights[2] * moments_sr3_2[2] +
            nonlinear_weights[3] * moments_sr3_3[2],
            nonlinear_weights[0] / linear_weights[0] * moments_sr5[3],
            nonlinear_weights[0] / linear_weights[0] * moments_sr5[4]
        ])

        polys_at_plus_half = np.asarray(
            [1.0, 0.5, 0.16666666666666666, 0.05, 0.014285714285714289])
        polys_at_minus_half = np.asarray(
            [1.0, -0.5, 0.16666666666666666, -0.05, 0.014285714285714289])
        return [
            np.sum(moments * polys_at_minus_half),
            np.sum(moments * polys_at_plus_half)
        ]

    def compute_face_values(recons_upper_of_cell, recons_lower_of_cell, v, i,
                            j, k, dim_to_recons):
        if dim_to_recons == 0:
            lower, upper = aoweno53(
                np.asarray([
                    v[i - 2, j, k], v[i - 1, j, k], v[i, j, k], v[i + 1, j, k],
                    v[i + 2, j, k]
                ]))
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)
        if dim_to_recons == 1:
            lower, upper = aoweno53(
                np.asarray([
                    v[i, j - 2, k], v[i, j - 1, k], v[i, j, k], v[i, j + 1, k],
                    v[i, j + 2, k]
                ]))
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)
        if dim_to_recons == 2:
            lower, upper = aoweno53(
                np.asarray([
                    v[i, j, k - 2], v[i, j, k - 1], v[i, j, k], v[i, j, k + 1],
                    v[i, j, k + 2]
                ]))
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)

    return Reconstruction.reconstruct(u, extents, dim, [2, 2, 2],
                                      compute_face_values)
