# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def christoffel_first_kind(d_metric):
    dim = d_metric.shape[0]
    return 0.5 * np.array([[[d_metric[b, c, a] + d_metric[a, c, b] - d_metric[
        c, a, b] for b in range(dim)] for a in range(dim)] for c in range(dim)])
