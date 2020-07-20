# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import sqrt, exp
from scipy.optimize import newton

# Isotropic Schwarzschild coordinates


def conformal_spatial_metric_isotropic(x, mass):
    return np.identity(3)


def extrinsic_curvature_trace_isotropic(x, mass):
    return 0.


def extrinsic_curvature_trace_gradient_isotropic(x, mass):
    return np.zeros(3)


def conformal_factor_isotropic(x, mass):
    r = np.linalg.norm(x)
    return 1. + 0.5 * mass / r


def conformal_factor_gradient_isotropic(x, mass):
    r = np.linalg.norm(x)
    return -0.5 * mass * x / r**3


def lapse_times_conformal_factor_isotropic(x, mass):
    r = np.linalg.norm(x)
    return 1. - 0.5 * mass / r


def lapse_times_conformal_factor_gradient_isotropic(x, mass):
    r = np.linalg.norm(x)
    return 0.5 * mass * x / r**3


def shift_background(x, mass):
    return np.zeros(3)


def shift_isotropic(x, mass):
    return np.zeros(3)


def shift_strain_isotropic(x, mass):
    return np.zeros((3, 3))


# Matter sources


def energy_density(x, mass):
    return 0.


def stress_trace(x, mass):
    return 0.


def momentum_density(x, mass):
    return np.zeros(3)


# Fixed sources


def conformal_factor_fixed_source(x, mass):
    return 0.


def lapse_times_conformal_factor_fixed_source(x, mass):
    return 0.


def shift_fixed_source(x, mass):
    return np.zeros(3)
