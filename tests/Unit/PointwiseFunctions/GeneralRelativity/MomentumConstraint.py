# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def momentum_constraint_in_vacuum(
    d_extrinsic_curvature, d_trace_extrinsic_curvature, inverse_spatial_metric
):
    return (
        np.einsum("jk,jki->i", inverse_spatial_metric, d_extrinsic_curvature)
        - d_trace_extrinsic_curvature
    )
