# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

x_coords = [-5.0, 6.0]
y_offset = 0.02
z_offset = 0.01


def shift(spacetime_metric):
    spatial_metric = spacetime_metric[1:, 1:]
    inverse_spatial_metric = np.linalg.inv(spatial_metric)
    g_jt = spacetime_metric[1:, 0]

    shift = np.dot(inverse_spatial_metric, g_jt)

    return shift


def lapse(spacetime_metric):
    spatial_metric = spacetime_metric[1:, 1:]
    inverse_spatial_metric = np.linalg.inv(spatial_metric)
    g_jt = spacetime_metric[1:, 0]
    shift = np.dot(inverse_spatial_metric, g_jt)
    beta_g_it = np.dot(shift, g_jt)
    g_tt = spacetime_metric[0, 0]

    lapse = np.sqrt(beta_g_it - g_tt)

    return lapse


def conformal_metric_bbh_isotropic(x):
    return np.identity(3)


def inv_conformal_metric_bbh_isotropic(x):
    return np.identity(3)


def shift_background(x):
    return np.zeros(3)


def longitudinal_shift_background_bbh_isotropic(x):
    return np.zeros((3, 3))


def conformal_factor_minus_one_bbh_isotropic(x):
    return 0.0


def energy_density_bbh_isotropic(x):
    return 0.0


def stress_trace_bbh_isotropic(x):
    return 0.0


def momentum_density_bbh_isotropic(x):
    return np.zeros(3)
