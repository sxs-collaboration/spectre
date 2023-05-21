# Distributed under the MIT License.
# See LICENSE.txt for details.

import Minmod  # for minmod() function with 2 args
import numpy as np
import Reconstruction
from numpy import asarray, sign


def mp5(v):
    # define the minmod() function with 4 args
    def minmod_4(a, b, c, d):
        return (
            0.125
            * (sign(a) + sign(b))
            * np.abs((sign(a) + sign(c)) * (sign(a) + sign(d)))
            * np.min(np.abs([a, b, c, d]))
        )

    def mp5_oneside(v, s):
        alpha = 4.0
        eps = 1e-10

        j = 2

        vor = (
            3.0 * v[j - 2 * s]
            - 20.0 * v[j - s]
            + 90.0 * v[j]
            + 60.0 * v[j + s]
            - 5.0 * v[j + 2 * s]
        ) / 128.0
        vmp = v[j] + Minmod.minmod(v[j + s] - v[j], alpha * (v[j] - v[j - s]))

        if (vor - v[j]) * (vor - vmp) > eps:
            djm1 = v[j - 2 * s] - 2.0 * v[j - s] + v[j]
            dj = v[j - s] - 2.0 * v[j] + v[j + s]
            djp1 = v[j] - 2.0 * v[j + s] + v[j + 2 * s]
            dm4jph = minmod_4(4.0 * dj - djp1, 4.0 * djp1 - dj, dj, djp1)
            dm4jmh = minmod_4(4.0 * dj - djm1, 4.0 * djm1 - dj, dj, djm1)

            vul = v[j] + alpha * (v[j] - v[j - s])
            vav = 0.5 * (v[j] + v[j + s])
            vmd = vav - 0.5 * dm4jph
            vlc = v[j] + 0.5 * (v[j] - v[j - s]) + (4.0 / 3.0) * dm4jmh

            vmin = max(np.min([v[j], v[j + s], vmd]), np.min([v[j], vul, vlc]))
            vmax = min(np.max([v[j], v[j + s], vmd]), np.max([v[j], vul, vlc]))

            return vor + Minmod.minmod(vmin - vor, vmax - vor)
        else:
            return vor

    return [mp5_oneside(v, -1), mp5_oneside(v, 1)]


def test_mp5(u, extents, dim):
    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        if dim_to_recons == 0:
            lower, upper = mp5(
                asarray(
                    [
                        v[i - 2, j, k],
                        v[i - 1, j, k],
                        v[i, j, k],
                        v[i + 1, j, k],
                        v[i + 2, j, k],
                    ]
                )
            )
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)
        if dim_to_recons == 1:
            lower, upper = mp5(
                asarray(
                    [
                        v[i, j - 2, k],
                        v[i, j - 1, k],
                        v[i, j, k],
                        v[i, j + 1, k],
                        v[i, j + 2, k],
                    ]
                )
            )
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)
        if dim_to_recons == 2:
            lower, upper = mp5(
                asarray(
                    [
                        v[i, j, k - 2],
                        v[i, j, k - 1],
                        v[i, j, k],
                        v[i, j, k + 1],
                        v[i, j, k + 2],
                    ]
                )
            )
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)

    return Reconstruction.reconstruct(
        u, extents, dim, [2, 2, 2], compute_face_values
    )
