# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import Reconstruction
import MonotonisedCentral
import Wcns5z

import scipy.signal


def test_wcns5z_mc(u, extents, dim):
    def wcns5z_mc(q):
        j = 2

        # indices of local maxima
        idx_local_max = scipy.signal.argrelextrema(q, np.greater)[0]
        # indices of local minima
        idx_local_min = scipy.signal.argrelextrema(q, np.less)[0]

        # number of local maxima and minima
        n_extrema = len(idx_local_max) + len(idx_local_min)

        if (n_extrema <= 1):
            return Wcns5z.wcns5z(q)
        else:
            slope = MonotonisedCentral.monotonised_central(
                q[j] - q[j - 1], q[j + 1] - q[j])
            return [q[j] - 0.5 * slope, q[j] + 0.5 * slope]

    def compute_face_values(recons_upper_of_cell, recons_lower_of_cell, v, i,
                            j, k, dim_to_recons):
        if dim_to_recons == 0:
            lower, upper = wcns5z_mc(
                np.asarray([
                    v[i - 2, j, k], v[i - 1, j, k], v[i, j, k], v[i + 1, j, k],
                    v[i + 2, j, k]
                ]))
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)
        if dim_to_recons == 1:
            lower, upper = wcns5z_mc(
                np.asarray([
                    v[i, j - 2, k], v[i, j - 1, k], v[i, j, k], v[i, j + 1, k],
                    v[i, j + 2, k]
                ]))
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)
        if dim_to_recons == 2:
            lower, upper = wcns5z_mc(
                np.asarray([
                    v[i, j, k - 2], v[i, j, k - 1], v[i, j, k], v[i, j, k + 1],
                    v[i, j, k + 2]
                ]))
            recons_lower_of_cell.append(lower)
            recons_upper_of_cell.append(upper)

    return Reconstruction.reconstruct(u, extents, dim, [2, 2, 2],
                                      compute_face_values)
