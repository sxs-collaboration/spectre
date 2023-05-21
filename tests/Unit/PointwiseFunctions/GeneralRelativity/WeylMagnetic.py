# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def weyl_magnetic_tensor(
    grad_extrinsic_curvature, spatial_metric, sqrt_det_spatial_metric
):
    levi_civita = np.zeros((3, 3, 3))
    levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1.0
    levi_civita[0, 2, 1] = levi_civita[2, 1, 0] = levi_civita[1, 0, 2] = -1.0

    det_spatial_metric = np.linalg.det(spatial_metric)
    result = (0.5 / sqrt_det_spatial_metric) * (
        np.einsum(
            "kli,mlk,jm", grad_extrinsic_curvature, levi_civita, spatial_metric
        )
        + np.einsum(
            "klj,mlk,im", grad_extrinsic_curvature, levi_civita, spatial_metric
        )
    )
    return result
