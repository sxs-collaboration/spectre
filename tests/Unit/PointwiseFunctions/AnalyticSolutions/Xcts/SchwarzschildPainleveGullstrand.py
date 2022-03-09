# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import sqrt


def conformal_metric(x, mass):
    return np.identity(3)


def inv_conformal_metric(x, mass):
    return np.identity(3)


def deriv_conformal_metric(x, mass):
    return np.zeros((3, 3, 3))


def extrinsic_curvature_trace(x, mass):
    r = np.linalg.norm(x)
    return 3. / 2. * sqrt(2. * mass / r**3)


def extrinsic_curvature_trace_gradient(x, mass):
    r = np.linalg.norm(x)
    return -9. / 4. * sqrt(2. * mass / r**5) * x / r


def conformal_factor(x, mass):
    return 1.


def conformal_factor_gradient(x, mass):
    return np.zeros(3)


def lapse_times_conformal_factor(x, mass):
    return 1.


def lapse_times_conformal_factor_gradient(x, mass):
    return np.zeros(3)


def lapse(x, mass):
    return 1.


def shift_background(x, mass):
    return np.zeros(3)


def longitudinal_shift_background_minus_dt_conformal_metric(x, mass):
    return np.zeros((3, 3))


def shift(x, mass):
    r = np.linalg.norm(x)
    return sqrt(2. * mass / r) * x / r


def shift_strain(x, mass):
    r = np.linalg.norm(x)
    return (sqrt(2. * mass / r**3) *
            (np.identity(3) - 3. / 2. * np.outer(x, x) / r**2))


def longitudinal_shift(x, mass):
    B = shift_strain(x, mass)
    return 2 * (B - 1. / 3. * np.identity(3) * np.trace(B))


def shift_dot_extrinsic_curvature_trace_gradient(x, mass):
    beta = shift(x, mass)
    dK = extrinsic_curvature_trace_gradient(x, mass)
    return np.dot(beta, dK)


def longitudinal_shift_minus_dt_conformal_metric_square(x, mass):
    LB = longitudinal_shift(x, mass)
    return np.einsum('ij,ij', LB, LB)


def longitudinal_shift_minus_dt_conformal_metric_over_lapse_square(x, mass):
    LB = longitudinal_shift(x, mass)
    return np.einsum('ij,ij', LB, LB) / lapse(x, mass)**2


# Matter sources


def energy_density(x, mass):
    return 0.


def stress_trace(x, mass):
    return 0.


def momentum_density(x, mass):
    return np.zeros(3)
