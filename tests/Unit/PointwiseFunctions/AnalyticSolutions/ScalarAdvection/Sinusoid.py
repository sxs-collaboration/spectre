# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def u(x, t):
    return np.sin(np.pi * (x[0] - t))
