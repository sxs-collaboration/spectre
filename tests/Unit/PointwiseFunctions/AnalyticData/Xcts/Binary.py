# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

centers = [-5., 6.]
masses = [1.1, 0.43]
angular_velocity = 0.02
falloff_widths = [7., 8.]


def conformal_metric_bbh_isotropic(x):
    return np.identity(3)


def inv_conformal_metric_bbh_isotropic(x):
    return np.identity(3)


def deriv_conformal_metric_bbh_isotropic(x):
    return np.zeros((3, 3, 3))


def extrinsic_curvature_trace_bbh_isotropic(x):
    return 0.


def shift_background(x):
    return np.array([-angular_velocity * x[1], angular_velocity * x[0], 0.])


def longitudinal_shift_background_bbh_isotropic(x):
    return np.zeros((3, 3))


def conformal_factor_bbh_isotropic(x):
    r1 = np.sqrt((x[0] - centers[0])**2 + x[1]**2 + x[2]**2)
    r2 = np.sqrt((x[0] - centers[1])**2 + x[1]**2 + x[2]**2)
    return 1. + 0.5 * (np.exp(-r1**2 / falloff_widths[0]**2) * masses[0] / r1 +
                       np.exp(-r2**2 / falloff_widths[1]**2) * masses[1] / r2)
