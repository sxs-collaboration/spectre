# Distributed under the MIT License.
# See LICENSE.txt for details.

import Minmod
import MonotonisedCentral
import numpy as np
import Reconstruction
import scipy.signal


def wcns5z(q):
    def wcns5z_oneside(q, s):
        j = 2

        b0 = (
            0.25 * (q[j - 2 * s] - 4.0 * q[j - s] + 3.0 * q[j]) ** 2
            + (13.0 / 12.0) * (q[j - 2 * s] - 2.0 * q[j - s] + q[j]) ** 2
        )
        b1 = (
            0.25 * (q[j + s] - q[j - s]) ** 2
            + (13.0 / 12.0) * (q[j - s] + q[j + s] - 2.0 * q[j]) ** 2
        )
        b2 = (
            0.25 * (q[j + 2 * s] - 4.0 * q[j + s] + 3.0 * q[j]) ** 2
            + (13.0 / 12.0) * (q[j + 2 * s] - 2.0 * q[j + s] + q[j]) ** 2
        )

        e0 = 2e-16 * (1.0 + abs(q[j - 2 * s]) + abs(q[j - s]) + abs(q[j]))
        e1 = 2e-16 * (1.0 + abs(q[j - s]) + abs(q[j]) + abs(q[j + s]))
        e2 = 2e-16 * (1.0 + abs(q[j + 2 * s]) + abs(q[j + s]) + abs(q[j]))

        beta = np.asarray([b0, b1, b2])
        epsilon = np.asarray([e0, e1, e2])
        tau5 = np.abs(b2 - b0)

        alpha = (
            np.asarray([1, 10, 5])
            / 16.0
            * (1.0 + (tau5 / (beta + epsilon)) ** 2)
        )
        nw = alpha / np.sum(alpha)

        recons_stencils = (
            np.asarray(
                [
                    3.0 * q[j - 2 * s] - 10.0 * q[j - s] + 15.0 * q[j],
                    -q[j - s] + 6.0 * q[j] + 3.0 * q[j + s],
                    3.0 * q[j] + 6.0 * q[j + s] - q[j + 2 * s],
                ]
            )
            / 8.0
        )

        return np.sum(nw * recons_stencils)

    return [wcns5z_oneside(q, -1), wcns5z_oneside(q, 1)]


def compute_face_values_t(
    recons_upper_of_cell,
    recons_lower_of_cell,
    v,
    i,
    j,
    k,
    dim_to_recons,
    wcns5z_type,
):
    if dim_to_recons == 0:
        lower, upper = wcns5z_type(
            np.asarray(
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
        lower, upper = wcns5z_type(
            np.asarray(
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
        lower, upper = wcns5z_type(
            np.asarray(
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


def test_wcns5z(u, extents, dim):
    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        return compute_face_values_t(
            recons_upper_of_cell,
            recons_lower_of_cell,
            v,
            i,
            j,
            k,
            dim_to_recons,
            wcns5z,
        )

    return Reconstruction.reconstruct(
        u, extents, dim, [2, 2, 2], compute_face_values
    )


def test_wcns5z_with_minmod(u, extents, dim):
    def wcns5z_with_minmod(q):
        j = 2

        # indices of local maxima
        idx_local_max = scipy.signal.argrelextrema(q, np.greater)[0]
        # indices of local minima
        idx_local_min = scipy.signal.argrelextrema(q, np.less)[0]
        # number of local maxima and minima
        n_extrema = len(idx_local_max) + len(idx_local_min)

        if n_extrema <= 1:
            return wcns5z(q)
        else:
            slope = Minmod.minmod(q[j] - q[j - 1], q[j + 1] - q[j])
            return [q[j] - 0.5 * slope, q[j] + 0.5 * slope]

    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        return compute_face_values_t(
            recons_upper_of_cell,
            recons_lower_of_cell,
            v,
            i,
            j,
            k,
            dim_to_recons,
            wcns5z_with_minmod,
        )

    return Reconstruction.reconstruct(
        u, extents, dim, [2, 2, 2], compute_face_values
    )


def test_wcns5z_with_mc(u, extents, dim):
    def wcns5z_with_mc(q):
        j = 2

        # indices of local maxima
        idx_local_max = scipy.signal.argrelextrema(q, np.greater)[0]
        # indices of local minima
        idx_local_min = scipy.signal.argrelextrema(q, np.less)[0]
        # number of local maxima and minima
        n_extrema = len(idx_local_max) + len(idx_local_min)

        if n_extrema <= 1:
            return wcns5z(q)
        else:
            slope = MonotonisedCentral.monotonised_central(
                q[j] - q[j - 1], q[j + 1] - q[j]
            )
            return [q[j] - 0.5 * slope, q[j] + 0.5 * slope]

    def compute_face_values(
        recons_upper_of_cell, recons_lower_of_cell, v, i, j, k, dim_to_recons
    ):
        return compute_face_values_t(
            recons_upper_of_cell,
            recons_lower_of_cell,
            v,
            i,
            j,
            k,
            dim_to_recons,
            wcns5z_with_mc,
        )

    return Reconstruction.reconstruct(
        u, extents, dim, [2, 2, 2], compute_face_values
    )
