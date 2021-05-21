# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def constraint_preserving_bjorhus_corrections_dt_v_psi(
    unit_interface_normal_vector, three_index_constraint, char_speeds):
    return (char_speeds[0] * np.einsum(
        'i,iab->ab', unit_interface_normal_vector, three_index_constraint))
