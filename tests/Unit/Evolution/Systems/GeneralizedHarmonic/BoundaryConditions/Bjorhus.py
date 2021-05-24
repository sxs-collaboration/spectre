# Distributed under the MIT License.
# See LICENSE.txt for details.

import itertools as it
import numpy as np


def constraint_preserving_bjorhus_corrections_dt_v_psi(
    unit_interface_normal_vector, three_index_constraint, char_speeds):
    return (char_speeds[0] * np.einsum(
        'i,iab->ab', unit_interface_normal_vector, three_index_constraint))


def constraint_preserving_bjorhus_corrections_dt_v_zero(
    unit_interface_normal_vector, four_index_constraint, char_speeds):
    spatial_dim = len(unit_interface_normal_vector)
    result = np.zeros([spatial_dim, 1 + spatial_dim, 1 + spatial_dim])

    if spatial_dim == 2:
        result[0, :, :] += char_speeds[1] * unit_interface_normal_vector[
            1] * four_index_constraint[1, :, :]
        result[1, :, :] += char_speeds[1] * unit_interface_normal_vector[
            0] * four_index_constraint[0, :, :]
    elif spatial_dim == 3:

        def is_even(sequence):
            count = 0
            for i, n in enumerate(sequence, start=1):
                count += sum(n > num for num in sequence[i:])
            return not count % 2

        for p in it.permutations(np.arange(len(unit_interface_normal_vector))):
            sgn = 1 if is_even(p) else -1
            result[p[0], :, :] += (sgn * char_speeds[1] *
                                   unit_interface_normal_vector[p[2]] *
                                   four_index_constraint[p[1], :, :])
    return result
