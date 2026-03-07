# Distributed under the MIT License.
# See LICENSE.txt for details.

import Reconstruction


def ppm(u_l_in, u_j, u_r_in):
    # Step 1: Unlimited quadratic (degree-2) interpolation
    u_L = (0.375 * u_l_in) + (0.75 * u_j) - (0.125 * u_r_in)
    u_R = -(0.125 * u_l_in) + (0.75 * u_j) + (0.375 * u_r_in)

    # Step 2: Colella & Woodward monotonicity limiter
    # Local extremum check: if cell is extremum, return cell center values
    if (u_R - u_j) * (u_j - u_L) <= 0.0:
        return (u_j, u_j)

    # Apply C&W limiter to prevent overshoot
    delta = u_R - u_L
    u_6 = 6.0 * (u_j - 0.5 * (u_L + u_R))

    if delta * (delta - u_6) < 0.0:
        u_L = (3.0 * u_j) - (2.0 * u_R)
    if delta * (delta + u_6) < 0.0:
        u_R = (3.0 * u_j) - (2.0 * u_L)

    return (u_L, u_R)


def test_ppm(u, extents, dim):
    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        if dim_to_recons == 0:
            (u_l, u_r) = ppm(v[i - 1, j, k], v[i, j, k], v[i + 1, j, k])
            recons_lower_of_cell.append(u_l)
            recons_upper_of_cell.append(u_r)
        if dim_to_recons == 1:
            (u_l, u_r) = ppm(v[i, j - 1, k], v[i, j, k], v[i, j + 1, k])
            recons_lower_of_cell.append(u_l)
            recons_upper_of_cell.append(u_r)
        if dim_to_recons == 2:
            (u_l, u_r) = ppm(v[i, j, k - 1], v[i, j, k], v[i, j, k + 1])
            recons_lower_of_cell.append(u_l)
            recons_upper_of_cell.append(u_r)

    return Reconstruction.reconstruct(
        u, extents, dim, [1, 1, 1], compute_face_values
    )
