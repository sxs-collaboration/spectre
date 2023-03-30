# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def weyl_type_D1_scalar(weyl_type_D1_tensor, inverse_spatial_metric):
    return np.einsum(
        "ij,kl,lj,ki",
        weyl_type_D1_tensor,
        weyl_type_D1_tensor,
        inverse_spatial_metric,
        inverse_spatial_metric,
    )
