# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def stress(strain, x, c_11, c_12, c_44):
    return -(c_11 - c_12 - 2 * c_44) * np.diag(np.diag(strain)) \
        -c_12 * np.identity(3) * np.trace(strain) \
        -2. * c_44 * strain
