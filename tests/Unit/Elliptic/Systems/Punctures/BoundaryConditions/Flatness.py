# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def normal_dot_field_gradient(field, x):
    return -field / np.linalg.norm(x)
