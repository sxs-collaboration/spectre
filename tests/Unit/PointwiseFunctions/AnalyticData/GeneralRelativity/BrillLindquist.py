# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def x_minus_center_a(x, center_a):
    return x - center_a


def r_a(x, center_a):
    x_a = x_minus_center_a(x, center_a)
    return np.sqrt(np.dot(x_a, x_a))


def x_minus_center_b(x, center_b):
    return x - center_b


def r_b(x, center_b):
    x_b = x_minus_center_b(x, center_b)
    return np.sqrt(np.dot(x_b, x_b))


def conformal_factor(x, mass_a, mass_b, center_a, center_b):
    return (
        1.0 + 0.5 * mass_a / r_a(x, center_a) + 0.5 * mass_b / r_b(x, center_b)
    )


def deriv_conformal_factor(x, mass_a, mass_b, center_a, center_b):
    return -0.5 * mass_a * x_minus_center_a(x, center_a) / np.power(
        r_a(x, center_a), 3
    ) - 0.5 * mass_b * x_minus_center_b(x, center_b) / np.power(
        r_b(x, center_b), 3
    )


def spatial_metric(x, mass_a, mass_b, center_a, center_b):
    spatial_metric = np.diag(np.ones_like(x))
    return spatial_metric * np.power(
        conformal_factor(x, mass_a, mass_b, center_a, center_b), 4
    )


def d_spatial_metric(x, mass_a, mass_b, center_a, center_b):
    d_spatial_metric = np.zeros((len(x), len(x), len(x)))
    four_psi_cubed = 4.0 * np.power(
        conformal_factor(x, mass_a, mass_b, center_a, center_b), 3
    )
    d_psi = deriv_conformal_factor(x, mass_a, mass_b, center_a, center_b)
    d_spatial_metric[:, 0, 0] = four_psi_cubed * d_psi
    d_spatial_metric[:, 1, 1] = four_psi_cubed * d_psi
    d_spatial_metric[:, 2, 2] = four_psi_cubed * d_psi
    return d_spatial_metric


def dt_spatial_metric(x, mass_a, mass_b, center_a, center_b):
    return np.zeros((len(x), len(x)))


def lapse(x, mass_a, mass_b, center_a, center_b):
    return 1.0


def dt_lapse(x, mass_a, mass_b, center_a, center_b):
    return 0.0


def d_lapse(x, mass_a, mass_b, center_a, center_b):
    return np.zeros_like(x)


def shift(x, mass_a, mass_b, center_a, center_b):
    return np.zeros_like(x)


def dt_shift(x, mass_a, mass_b, center_a, center_b):
    return np.zeros_like(x)


def d_shift(x, mass_a, mass_b, center_a, center_b):
    return np.zeros((len(x), len(x)))


def sqrt_det_spatial_metric(x, mass_a, mass_b, center_a, center_b):
    return np.sqrt(
        np.linalg.det(spatial_metric(x, mass_a, mass_b, center_a, center_b))
    )


def extrinsic_curvature(x, mass_a, mass_b, center_a, center_b):
    return np.zeros((len(x), len(x)))


def inverse_spatial_metric(x, mass_a, mass_b, center_a, center_b):
    return np.linalg.inv(spatial_metric(x, mass_a, mass_b, center_a, center_b))
