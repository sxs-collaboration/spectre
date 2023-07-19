# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from PointwiseFunctions.GeneralRelativity.ComputeGhQuantities import (
    deriv_lapse,
    dt_lapse,
    spacetime_deriv_detg,
)


def spatial_weight_function(coords, r_max):
    r2 = np.sum([coords[i] ** 2 for i in range(len(coords))])
    return np.exp(-r2 / r_max / r_max)


def spacetime_deriv_spatial_weight_function(coords, r_max, W=None):
    if W is None:
        W = spatial_weight_function(coords, r_max)
    DW = np.zeros(len(coords) + 1)
    DW[1:] = -2.0 * W / r_max / r_max * coords
    return DW


def log_fac(lapse, sqrt_det_spatial_metric, exponent):
    # if exponent == 0, the first term automatically vanishes
    return exponent * np.log(sqrt_det_spatial_metric**2) - np.log(lapse)
