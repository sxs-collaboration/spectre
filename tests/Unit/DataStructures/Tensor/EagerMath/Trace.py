# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def trace_last_indices(tensor, metric):
    return np.einsum("ij,kij", metric, tensor)
