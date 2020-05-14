# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def pi_upwind_penalty_correction(constraint_gamma2, v_psi_int, v_zero_int,
                                 v_plus_int, v_minus_int, char_speeds_int,
                                 unit_normal_int, v_psi_ext, v_zero_ext,
                                 v_plus_ext, v_minus_ext, char_speeds_ext,
                                 unit_normal_ext):
    result = v_psi_int * 0.

    # Add v^+ terms
    if char_speeds_ext[2] > 0.:
        result -= 0.5 * char_speeds_ext[2] * v_plus_ext
    if char_speeds_int[2] < 0.:
        result -= 0.5 * char_speeds_int[2] * v_plus_int

    # Add v^- terms
    if char_speeds_ext[3] > 0.:
        result -= 0.5 * char_speeds_ext[3] * v_minus_ext
    if char_speeds_int[3] < 0.:
        result -= 0.5 * char_speeds_int[3] * v_minus_int

    # Add v^\Psi terms
    if char_speeds_ext[0] > 0.:
        result -= char_speeds_ext[0] * constraint_gamma2 * v_psi_ext
    if char_speeds_int[0] < 0.:
        result -= char_speeds_int[0] * constraint_gamma2 * v_psi_int
    return result


def phi_upwind_penalty_correction(constraint_gamma2, v_psi_int, v_zero_int,
                                  v_plus_int, v_minus_int, char_speeds_int,
                                  unit_normal_int, v_psi_ext, v_zero_ext,
                                  v_plus_ext, v_minus_ext, char_speeds_ext,
                                  unit_normal_ext):
    result = v_zero_int * 0

    # Add v^+ terms
    if char_speeds_ext[2] > 0.:
        result -= 0.5 * char_speeds_ext[2] * v_plus_ext * unit_normal_ext
    if char_speeds_int[2] < 0.:
        result -= 0.5 * char_speeds_int[2] * v_plus_int * unit_normal_int

    # Add v^- terms
    if char_speeds_ext[3] > 0.:
        result += 0.5 * char_speeds_ext[3] * v_minus_ext * unit_normal_ext
    if char_speeds_int[3] < 0.:
        result += 0.5 * char_speeds_int[3] * v_minus_int * unit_normal_int

    # Add v^0 terms
    if char_speeds_ext[1] > 0.:
        result -= char_speeds_ext[1] * v_zero_ext
    if char_speeds_int[1] < 0.:
        result -= char_speeds_int[1] * v_zero_int
    return result


def psi_upwind_penalty_correction(constraint_gamma2, v_psi_int, v_zero_int,
                                  v_plus_int, v_minus_int, char_speeds_int,
                                  unit_normal_int, v_psi_ext, v_zero_ext,
                                  v_plus_ext, v_minus_ext, char_speeds_ext,
                                  unit_normal_ext):
    result = v_psi_int * 0
    if char_speeds_ext[0] > 0.:
        result -= char_speeds_ext[0] * v_psi_ext
    if char_speeds_int[0] < 0.:
        result -= char_speeds_int[0] * v_psi_int
    return result
