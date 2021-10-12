# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def a_tilde(conformal_factor_squared, spatial_metric, extrinsic_curvature,
            trace_extrinsic_curvature):
    return conformal_factor_squared * (
        extrinsic_curvature - trace_extrinsic_curvature * spatial_metric / 3.0)
