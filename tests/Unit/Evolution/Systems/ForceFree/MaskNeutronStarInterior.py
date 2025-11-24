# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def compute_ns_interior_mask(coords):
    r_squared = np.einsum("a, a", coords, coords)

    if r_squared < 1.0:
        return -1.0
    else:
        return 1.0
