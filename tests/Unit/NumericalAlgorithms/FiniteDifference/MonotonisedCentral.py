# Distributed under the MIT License.
# See LICENSE.txt for details.

import Reconstruction


def monotonised_central(a, b):
    sign = lambda x: -1 if x < 0 else 1
    sign_a = sign(a)
    sign_b = sign(b)
    return (
        0.5
        * (sign_a + sign_b)
        * min(0.5 * abs(a + b), min(2.0 * abs(a), 2.0 * abs(b)))
    )


def test_monotonised_central(u, extents, dim):
    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        if dim_to_recons == 0:
            slope = monotonised_central(
                v[i, j, k] - v[i - 1, j, k], v[i + 1, j, k] - v[i, j, k]
            )
            recons_lower_of_cell.append(v[i, j, k] - 0.5 * slope)
            recons_upper_of_cell.append(v[i, j, k] + 0.5 * slope)
        if dim_to_recons == 1:
            slope = monotonised_central(
                v[i, j, k] - v[i, j - 1, k], v[i, j + 1, k] - v[i, j, k]
            )
            recons_lower_of_cell.append(v[i, j, k] - 0.5 * slope)
            recons_upper_of_cell.append(v[i, j, k] + 0.5 * slope)
        if dim_to_recons == 2:
            slope = monotonised_central(
                v[i, j, k] - v[i, j, k - 1], v[i, j, k + 1] - v[i, j, k]
            )
            recons_lower_of_cell.append(v[i, j, k] - 0.5 * slope)
            recons_upper_of_cell.append(v[i, j, k] + 0.5 * slope)

    return Reconstruction.reconstruct(
        u, extents, dim, [1, 1, 1], compute_face_values
    )
