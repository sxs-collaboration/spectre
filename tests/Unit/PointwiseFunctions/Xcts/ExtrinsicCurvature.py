# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def extrinsic_curvature(conformal_factor, lapse, conformal_metric,
                        longitudinal_shift_minus_dt_conformal_metric,
                        trace_extrinsic_curvature):
    longitudinal_shift_minus_dt_conformal_metric_lower = np.einsum(
        'ij,ik,jl', longitudinal_shift_minus_dt_conformal_metric,
        conformal_metric, conformal_metric)
    return conformal_factor**4 * (
        longitudinal_shift_minus_dt_conformal_metric_lower / 2. / lapse +
        conformal_metric * trace_extrinsic_curvature / 3.)
