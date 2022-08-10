## Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import Reconstruction

import Minmod
import MonotonisedCentral


def _minmod(q):
    j = 1
    slope = Minmod.minmod(q[j] - q[j - 1], q[j + 1] - q[j])
    return [q[j] - 0.5 * slope, q[j] + 0.5 * slope]


def _mc(q):
    j = 1
    slope = MonotonisedCentral.monotonised_central(q[j] - q[j - 1],
                                                   q[j + 1] - q[j])
    return [q[j] - 0.5 * slope, q[j] + 0.5 * slope]


def _adaptive_order_5(q, keep_positive, low_order_recons):
    j = 2
    alpha_5 = 4.0
    norm_top = 0.2222222222222222 * (
        -1.4880952380952381 * q[j + 1] + 0.37202380952380953 * q[j + 2] -
        1.4880952380952381 * q[j - 1] + 0.37202380952380953 * q[j - 2] +
        2.232142857142857 * q[j])**2
    norm_full = (
        q[j + 1] *
        (1.179711612654321 * q[j + 1] - 0.963946414792769 * q[j + 2] +
         1.0904086750440918 * q[j - 1] - 0.5030502507716049 * q[j - 2] -
         1.6356130125661377 * q[j]) + q[j + 2] *
        (0.6699388830329586 * q[j + 2] - 0.5030502507716049 * q[j - 1] +
         0.154568572944224 * q[j - 2] + 0.927411437665344 * q[j]) + q[j - 1] *
        (1.179711612654321 * q[j - 1] - 0.963946414792769 * q[j - 2] -
         1.6356130125661377 * q[j]) + q[j - 2] *
        (0.6699388830329586 * q[j - 2] + 0.927411437665344 * q[j]) +
        1.4061182415674602 * q[j]**2)
    if (4.0**alpha_5 * norm_top <= norm_full):
        result = [
            -0.15625 * q[j + 1] + 0.0234375 * q[j + 2] + 0.46875 * q[j - 1] -
            0.0390625 * q[j - 2] + 0.703125 * q[j],
            0.46875 * q[j + 1] - 0.0390625 * q[j + 2] - 0.15625 * q[j - 1] +
            0.0234375 * q[j - 2] + 0.703125 * q[j]
        ]
        if (not keep_positive) or (result[0] > 0.0 and result[1] > 0.0):
            return result
    low_order_result = low_order_recons([q[j - 1], q[j], q[j + 1]])
    if (not keep_positive) or (low_order_result[0] > 0.0
                               and low_order_result[1] > 0.0):
        return low_order_result
    return [q[j], q[j]]


def compute_face_values_t(recons_upper_of_cell, recons_lower_of_cell, v, i, j,
                          k, dim_to_recons, keep_positive, low_order_recons):
    if dim_to_recons == 0:
        lower, upper = _adaptive_order_5(
            np.asarray([
                v[i - 2, j, k], v[i - 1, j, k], v[i, j, k], v[i + 1, j, k],
                v[i + 2, j, k]
            ]), keep_positive, low_order_recons)
        recons_lower_of_cell.append(lower)
        recons_upper_of_cell.append(upper)
    if dim_to_recons == 1:
        lower, upper = _adaptive_order_5(
            np.asarray([
                v[i, j - 2, k], v[i, j - 1, k], v[i, j, k], v[i, j + 1, k],
                v[i, j + 2, k]
            ]), keep_positive, low_order_recons)
        recons_lower_of_cell.append(lower)
        recons_upper_of_cell.append(upper)
    if dim_to_recons == 2:
        lower, upper = _adaptive_order_5(
            np.asarray([
                v[i, j, k - 2], v[i, j, k - 1], v[i, j, k], v[i, j, k + 1],
                v[i, j, k + 2]
            ]), keep_positive, low_order_recons)
        recons_lower_of_cell.append(lower)
        recons_upper_of_cell.append(upper)


def test_adaptive_order_with_mc(u, extents, dim):
    def compute_face_values(recons_upper_of_cell, recons_lower_of_cell, v, i,
                            j, k, dim_to_recons):
        return compute_face_values_t(recons_upper_of_cell,
                                     recons_lower_of_cell, v, i, j, k,
                                     dim_to_recons, False, _mc)

    return Reconstruction.reconstruct(u, extents, dim, [2, 2, 2],
                                      compute_face_values)


def test_adaptive_order_with_minmod(u, extents, dim):
    def compute_face_values(recons_upper_of_cell, recons_lower_of_cell, v, i,
                            j, k, dim_to_recons):
        return compute_face_values_t(recons_upper_of_cell,
                                     recons_lower_of_cell, v, i, j, k,
                                     dim_to_recons, False, _minmod)

    return Reconstruction.reconstruct(u, extents, dim, [2, 2, 2],
                                      compute_face_values)


def test_positivity_preserving_adaptive_order_with_mc(u, extents, dim):
    def compute_face_values(recons_upper_of_cell, recons_lower_of_cell, v, i,
                            j, k, dim_to_recons):
        return compute_face_values_t(recons_upper_of_cell,
                                     recons_lower_of_cell, v, i, j, k,
                                     dim_to_recons, True, _mc)

    return Reconstruction.reconstruct(u, extents, dim, [2, 2, 2],
                                      compute_face_values)


def test_positivity_preserving_adaptive_order_with_minmod(u, extents, dim):
    def compute_face_values(recons_upper_of_cell, recons_lower_of_cell, v, i,
                            j, k, dim_to_recons):
        return compute_face_values_t(recons_upper_of_cell,
                                     recons_lower_of_cell, v, i, j, k,
                                     dim_to_recons, True, _minmod)

    return Reconstruction.reconstruct(u, extents, dim, [2, 2, 2],
                                      compute_face_values)
