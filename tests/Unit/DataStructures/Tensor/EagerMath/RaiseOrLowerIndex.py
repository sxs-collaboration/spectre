# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def raise_or_lower_first_index(tensor, metric):
    return np.einsum("ij,ikl", metric, tensor)
