# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Test functions for one-index constraint


def one_index_constraint(d_psi, phi):
    return d_psi - phi


# End test functions for one-index constraint

# Test functions for two-index constraint


def two_index_constraint(d_phi):
    return d_phi - np.transpose(d_phi)


# End test functions for two-index constraint
