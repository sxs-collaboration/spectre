# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def weyl_electric_tensor(spatial_ricci, extrinsic_curvature,
                         inverse_spatial_metric):
    return (np.einsum("ij", spatial_ricci) +
            np.einsum("kl,kl,ij", extrinsic_curvature, inverse_spatial_metric,
                      extrinsic_curvature) -
            np.einsum("il,kl,kj", extrinsic_curvature, inverse_spatial_metric,
                      extrinsic_curvature))
