# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import exp, sqrt
from scipy.optimize import newton

# Areal Kerr-Schild coordinates


def lapse_areal(r, mass):
    return 1.0 / sqrt(1.0 + 2.0 * mass / r)


def lapse_areal_deriv(r, mass):
    return lapse_areal(r, mass) ** 3 * mass / r**2


def shift_magnitude_areal(r, mass):
    return 2.0 * mass * lapse_areal(r, mass) ** 2 / r


def shift_areal(x, mass):
    r = np.linalg.norm(x)
    return shift_magnitude_areal(r, mass) * x / r


def spatial_metric_areal(x, mass):
    r = np.linalg.norm(x)
    return np.identity(3) + 2.0 * mass * np.outer(x, x) / r**3


def extrinsic_curvature_trace_areal(r, mass):
    lapse = lapse_areal(r, mass)
    return 2.0 * mass * lapse**3 / r**2 * (1.0 + 3.0 * mass / r)


def extrinsic_curvature_trace_areal_deriv(r, mass):
    lapse = lapse_areal(r, mass)
    K = extrinsic_curvature_trace_areal(r, mass)
    return (
        K
        / r
        * (3.0 * mass * lapse**2 / r - 3.0 * mass / (r + 3.0 * mass) - 2.0)
    )


def extrinsic_curvature_areal(x, mass):
    r = np.linalg.norm(x)
    lapse = lapse_areal(r, mass)
    return (
        2.0
        * mass
        * lapse
        / r**2
        * (np.identity(3) - (2.0 + mass / r) * np.outer(x, x) / r**2)
    )


# Isotropic Kerr-Schild coordinates
# See e.g. Sec. 7.4.1 in https://arxiv.org/abs/gr-qc/0510016


def isotropic_radius_from_areal(r_areal, mass):
    one_over_lapse = 1.0 / lapse_areal(r_areal, mass)
    return (
        r_areal
        / 4.0
        * (1.0 + one_over_lapse) ** 2
        * exp(2.0 - 2.0 * one_over_lapse)
    )


def isotropic_radius_from_areal_deriv(r_areal, mass):
    r_isotropic = isotropic_radius_from_areal(r_areal, mass)
    return r_isotropic / r_areal / lapse_areal(r_areal, mass)


def areal_radius_from_isotropic(r_isotropic, mass):
    def f(r_areal):
        return isotropic_radius_from_areal(r_areal, mass) - r_isotropic

    def fprime(r_areal):
        return isotropic_radius_from_areal_deriv(r_areal, mass)

    return newton(func=f, fprime=fprime, x0=r_isotropic, tol=1.0e-12)


def conformal_metric(x, mass):
    return np.identity(3)


def inv_conformal_metric(x, mass):
    return np.identity(3)


def deriv_conformal_metric(x, mass):
    return np.zeros((3, 3, 3))


def extrinsic_curvature_trace(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return extrinsic_curvature_trace_areal(r_areal, mass)


def extrinsic_curvature_trace_gradient(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return (
        extrinsic_curvature_trace_areal_deriv(r_areal, mass)
        / isotropic_radius_from_areal_deriv(r_areal, mass)
        * x
        / r_isotropic
    )


def conformal_factor_from_areal(r_areal, mass):
    one_over_lapse = 1.0 / lapse_areal(r_areal, mass)
    return 2.0 * exp(one_over_lapse - 1.0) / (1.0 + one_over_lapse)


def conformal_factor_from_areal_deriv(r_areal, mass):
    one_over_lapse = 1.0 / lapse_areal(r_areal, mass)
    return (
        -2.0
        * mass
        * exp(one_over_lapse - 1.0)
        / (1.0 + one_over_lapse) ** 2
        / r_areal**2
    )


def conformal_factor(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return conformal_factor_from_areal(r_areal, mass)


def conformal_factor_gradient(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return (
        conformal_factor_from_areal_deriv(r_areal, mass)
        / isotropic_radius_from_areal_deriv(r_areal, mass)
        * x
        / r_isotropic
    )


def lapse_times_conformal_factor(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return lapse_areal(r_areal, mass) * conformal_factor_from_areal(
        r_areal, mass
    )


def lapse_times_conformal_factor_gradient(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return (
        (
            lapse_areal_deriv(r_areal, mass)
            * conformal_factor_from_areal(r_areal, mass)
            + lapse_areal(r_areal, mass)
            * conformal_factor_from_areal_deriv(r_areal, mass)
        )
        / isotropic_radius_from_areal_deriv(r_areal, mass)
        * x
        / r_isotropic
    )


def lapse(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return lapse_areal(r_areal, mass)


def shift_background(x, mass):
    return np.zeros(3)


def longitudinal_shift_background_minus_dt_conformal_metric(x, mass):
    return np.zeros((3, 3))


def shift_magnitude_from_areal(r_areal, mass):
    return (
        shift_magnitude_areal(r_areal, mass)
        / lapse_areal(r_areal, mass)
        / conformal_factor_from_areal(r_areal, mass) ** 2
    )


def shift(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return shift_magnitude_from_areal(r_areal, mass) * x / r_isotropic


def shift_strain(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    lapse = lapse_areal(r_areal, mass)
    return (
        2.0
        * mass
        * lapse
        / r_areal**2
        * (
            np.identity(3)
            + (mass * lapse**3 / r_areal - 2.0 * lapse)
            * np.outer(x, x)
            / r_isotropic**2
        )
    )


def longitudinal_shift(x, mass):
    B = shift_strain(x, mass)
    return 2 * (B - 1.0 / 3.0 * np.identity(3) * np.trace(B))


def shift_dot_extrinsic_curvature_trace_gradient(x, mass):
    beta = shift(x, mass)
    dK = extrinsic_curvature_trace_gradient(x, mass)
    return np.dot(beta, dK)


def longitudinal_shift_minus_dt_conformal_metric_square(x, mass):
    LB = longitudinal_shift(x, mass)
    return np.einsum("ij,ij", LB, LB)


def longitudinal_shift_minus_dt_conformal_metric_over_lapse_square(x, mass):
    LB = longitudinal_shift(x, mass)
    return np.einsum("ij,ij", LB, LB) / lapse(x, mass) ** 2


# Matter sources


def energy_density(x, mass):
    return 0.0


def stress_trace(x, mass):
    return 0.0


def momentum_density(x, mass):
    return np.zeros(3)
