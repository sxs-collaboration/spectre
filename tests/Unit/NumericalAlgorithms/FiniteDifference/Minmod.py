# Distributed under the MIT License.
# See LICENSE.txt for details.

import Reconstruction


def test_minmod(u, extents, dim):
    def minmod(a, b):
        if a * b <= 0.0:
            return 0.0
        if abs(a) < abs(b):
            return a
        return b

    def compute_face_values(recons_upper_of_cell, recons_lower_of_cell, v, i,
                            j, k, dim_to_recons):
        if dim_to_recons == 0:
            slope = minmod(v[i, j, k] - v[i - 1, j, k],
                           v[i + 1, j, k] - v[i, j, k])
            recons_lower_of_cell.append(v[i, j, k] - 0.5 * slope)
            recons_upper_of_cell.append(v[i, j, k] + 0.5 * slope)
        if dim_to_recons == 1:
            slope = minmod(v[i, j, k] - v[i, j - 1, k],
                           v[i, j + 1, k] - v[i, j, k])
            recons_lower_of_cell.append(v[i, j, k] - 0.5 * slope)
            recons_upper_of_cell.append(v[i, j, k] + 0.5 * slope)
        if dim_to_recons == 2:
            slope = minmod(v[i, j, k] - v[i, j, k - 1],
                           v[i, j, k + 1] - v[i, j, k])
            recons_lower_of_cell.append(v[i, j, k] - 0.5 * slope)
            recons_upper_of_cell.append(v[i, j, k] + 0.5 * slope)

    return Reconstruction.reconstruct(u, extents, dim, [1, 1, 1],
                                      compute_face_values)
