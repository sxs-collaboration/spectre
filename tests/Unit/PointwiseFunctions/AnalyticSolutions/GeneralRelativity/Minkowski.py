# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def minkowski_lapse(x, t):
    return 1.0


def minkowski_dt_lapse(x, t):
    return 0.0


def minkowski_d_lapse(x, t):
    return 0.0


def minkowski_shift(x, t):
    return np.zeros_like(x)


def minkowski_dt_shift(x, t):
    return np.zeros_like(x)


def minkowski_d_shift(x, t):
    return np.zeros((len(x), len(x)))


def minkowski_spatial_metric(x, t):
    spatial_metric = np.diag(np.ones_like(x))
    return spatial_metric


def minkowski_dt_spatial_metric(x, t):
    dt_spatial_metric = np.zeros((len(x), len(x)))
    return dt_spatial_metric


def minkowski_d_spatial_metric(x, t):
    d_spatial_metric = np.zeros((len(x), len(x), len(x)))
    return d_spatial_metric


def minkowski_sqrt_det_spatial_metric(x, t):
    return 1.0


def minkowski_extrinsic_curvature(x, t):
    extrinsic_curvature = np.zeros((len(x), len(x)))
    return extrinsic_curvature


def minkowski_inverse_spatial_metric(x, t):
    inverse_spatial_metric = np.diag(np.ones_like(x))
    return inverse_spatial_metric
