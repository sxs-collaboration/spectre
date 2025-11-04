# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import exp, sqrt
from scipy.optimize import newton

# Schwarzschild Maximal Isotropic coordinates


def isotropic_radius_from_areal(r_areal, mass):
    # First part of the equation
    part1 = (
        2 * r_areal
        + mass
        + sqrt(4 * r_areal**2 + 4 * mass * r_areal + 3 * mass**2)
    ) / 4.0

    # Second part of the equation
    part2_numerator = (4 + 3 * sqrt(2)) * (2 * r_areal - 3 * mass)
    part2_denominator = (
        8 * r_areal
        + 6 * mass
        + 3 * sqrt(8 * r_areal**2 + 8 * mass * r_areal + 6 * mass**2)
    )
    part2 = (part2_numerator / part2_denominator) ** (1 / sqrt(2))

    return part1 * part2


def isotropic_radius_from_areal_deriv(r_areal, mass):
    h = 1e-6
    term1 = -isotropic_radius_from_areal(r_areal + 2 * h, mass)
    term2 = 8 * isotropic_radius_from_areal(r_areal + h, mass)
    term3 = -8 * isotropic_radius_from_areal(r_areal - h, mass)
    term4 = isotropic_radius_from_areal(r_areal - 2 * h, mass)

    return (term1 + term2 + term3 + term4) / (12 * h)


def areal_radius_from_isotropic(r_isotropic, mass):
    def f(r_areal):
        return isotropic_radius_from_areal(r_areal, mass) - r_isotropic

    def fprime(r_areal):
        return isotropic_radius_from_areal_deriv(r_areal, mass)

    return newton(
        func=f,
        fprime=fprime,
        # Initial guess for r_areal can't be smaller than 1.5 * mass because
        # isotropic radius is not defined for smaller values.
        x0=max(r_isotropic, 2 * mass),
        tol=1.0e-12,
    )


def conformal_metric(x, mass):
    return np.identity(3)


def inv_conformal_metric(x, mass):
    return np.identity(3)


def deriv_conformal_metric(x, mass):
    return np.zeros((3, 3, 3))


def extrinsic_curvature_trace(x, mass):
    return 0.0


def extrinsic_curvature_trace_gradient(x, mass):
    return np.zeros(3)


def conformal_factor(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return sqrt(r_areal / r_isotropic)


def conformal_factor_gradient(x, mass):
    h = 1e-6

    x_forward_2h = np.array([x[0] + 2 * h, x[1], x[2]])
    x_forward_h = np.array([x[0] + h, x[1], x[2]])
    x_backward_h = np.array([x[0] - h, x[1], x[2]])
    x_backward_2h = np.array([x[0] - 2 * h, x[1], x[2]])
    dx = (
        1 / 12 * conformal_factor(x_backward_2h, mass)
        - 2 / 3 * conformal_factor(x_backward_h, mass)
        + 2 / 3 * conformal_factor(x_forward_h, mass)
        - 1 / 12 * conformal_factor(x_forward_2h, mass)
    ) / h

    y_forward_2h = np.array([x[0], x[1] + 2 * h, x[2]])
    y_forward_h = np.array([x[0], x[1] + h, x[2]])
    y_backward_h = np.array([x[0], x[1] - h, x[2]])
    y_backward_2h = np.array([x[0], x[1] - 2 * h, x[2]])
    dy = (
        1 / 12 * conformal_factor(y_backward_2h, mass)
        - 2 / 3 * conformal_factor(y_backward_h, mass)
        + 2 / 3 * conformal_factor(y_forward_h, mass)
        - 1 / 12 * conformal_factor(y_forward_2h, mass)
    ) / h

    z_forward_2h = np.array([x[0], x[1], x[2] + 2 * h])
    z_forward_h = np.array([x[0], x[1], x[2] + h])
    z_backward_h = np.array([x[0], x[1], x[2] - h])
    z_backward_2h = np.array([x[0], x[1], x[2] - 2 * h])
    dz = (
        1 / 12 * conformal_factor(z_backward_2h, mass)
        - 2 / 3 * conformal_factor(z_backward_h, mass)
        + 2 / 3 * conformal_factor(z_forward_h, mass)
        - 1 / 12 * conformal_factor(z_forward_2h, mass)
    ) / h

    return np.array([dx, dy, dz])


def lapse(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return sqrt(1 - 2 * mass / r_areal + (27 / 16) * mass**4 / r_areal**4)


def lapse_times_conformal_factor(x, mass):
    lapse_val = lapse(x, mass)
    conformal_factor_val = conformal_factor(x, mass)
    return lapse_val * conformal_factor_val


def lapse_times_conformal_factor_gradient(x, mass):
    h = 1e-6

    x_forward_2h = np.array([x[0] + 2 * h, x[1], x[2]])
    x_forward_h = np.array([x[0] + h, x[1], x[2]])
    x_backward_h = np.array([x[0] - h, x[1], x[2]])
    x_backward_2h = np.array([x[0] - 2 * h, x[1], x[2]])
    dx = (
        1 / 12 * lapse_times_conformal_factor(x_backward_2h, mass)
        - 2 / 3 * lapse_times_conformal_factor(x_backward_h, mass)
        + 2 / 3 * lapse_times_conformal_factor(x_forward_h, mass)
        - 1 / 12 * lapse_times_conformal_factor(x_forward_2h, mass)
    ) / h

    y_forward_2h = np.array([x[0], x[1] + 2 * h, x[2]])
    y_forward_h = np.array([x[0], x[1] + h, x[2]])
    y_backward_h = np.array([x[0], x[1] - h, x[2]])
    y_backward_2h = np.array([x[0], x[1] - 2 * h, x[2]])
    dy = (
        1 / 12 * lapse_times_conformal_factor(y_backward_2h, mass)
        - 2 / 3 * lapse_times_conformal_factor(y_backward_h, mass)
        + 2 / 3 * lapse_times_conformal_factor(y_forward_h, mass)
        - 1 / 12 * lapse_times_conformal_factor(y_forward_2h, mass)
    ) / h

    z_forward_2h = np.array([x[0], x[1], x[2] + 2 * h])
    z_forward_h = np.array([x[0], x[1], x[2] + h])
    z_backward_h = np.array([x[0], x[1], x[2] - h])
    z_backward_2h = np.array([x[0], x[1], x[2] - 2 * h])
    dz = (
        1 / 12 * lapse_times_conformal_factor(z_backward_2h, mass)
        - 2 / 3 * lapse_times_conformal_factor(z_backward_h, mass)
        + 2 / 3 * lapse_times_conformal_factor(z_forward_h, mass)
        - 1 / 12 * lapse_times_conformal_factor(z_forward_2h, mass)
    ) / h

    return np.array([dx, dy, dz])


def shift_background(x, mass):
    return np.zeros(3)


def longitudinal_shift_background_minus_dt_conformal_metric(x, mass):
    return np.zeros((3, 3))


def shift(x, mass):
    r_isotropic = np.linalg.norm(x)
    r_areal = areal_radius_from_isotropic(r_isotropic, mass)
    return (
        (0.75 * sqrt(3.0) * mass**2 * (r_isotropic / r_areal**3))
        * x
        / r_isotropic
    )


def shift_strain(x, mass):
    h = 1e-6
    n = len(x)
    strain_tensor = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_forward_2h = np.copy(x)
            x_forward_h = np.copy(x)
            x_backward_h = np.copy(x)
            x_backward_2h = np.copy(x)

            x_forward_2h[j] += 2 * h
            x_forward_h[j] += h
            x_backward_h[j] -= h
            x_backward_2h[j] -= 2 * h

            beta_forward_2h = shift(x_forward_2h, mass)[i]
            beta_forward_h = shift(x_forward_h, mass)[i]
            beta_backward_h = shift(x_backward_h, mass)[i]
            beta_backward_2h = shift(x_backward_2h, mass)[i]

            strain_tensor[i, j] = (
                1 / 12 * beta_backward_2h
                - 2 / 3 * beta_backward_h
                + 2 / 3 * beta_forward_h
                - 1 / 12 * beta_forward_2h
            ) / h

    return strain_tensor


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
