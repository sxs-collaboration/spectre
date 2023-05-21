# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def weyl_magnetic_scalar(weyl_magnetic, inverse_spatial_metric):
    return np.einsum(
        "ij,kl,jk,il",
        weyl_magnetic,
        weyl_magnetic,
        inverse_spatial_metric,
        inverse_spatial_metric,
    )
