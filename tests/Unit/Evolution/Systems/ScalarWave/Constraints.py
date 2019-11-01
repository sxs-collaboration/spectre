# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def one_index_constraint(dpsi, phi):
    return dpsi - phi


def two_index_constraint(dphi):
    return dphi - np.transpose(dphi)
