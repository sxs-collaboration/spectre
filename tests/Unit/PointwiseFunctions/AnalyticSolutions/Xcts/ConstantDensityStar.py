# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from scipy.optimize import newton


def compute_alpha(density, radius):
    def f(a):
        return (
            density * radius**2
            - 3.0 / (2.0 * np.pi) * a**10 / (1.0 + a**2) ** 6
        )

    def fprime(a):
        return 3.0 * a**9 * (a**2 - 5.0) / (1.0 + a**2) ** 7 / np.pi

    return newton(func=f, fprime=fprime, x0=2.0 * np.sqrt(5.0))


def sobolov(r, alpha, radius):
    return np.sqrt(alpha * radius / (r**2 + (alpha * radius) ** 2))


def conformal_factor(x, density, radius):
    alpha = compute_alpha(density, radius)
    C = (3.0 / (2.0 * np.pi * density)) ** (1.0 / 4.0)
    r = np.linalg.norm(x)
    if r <= radius:
        return C * sobolov(r, alpha, radius)
    else:
        beta = radius * (C * sobolov(radius, alpha, radius) - 1.0)
        return beta / r + 1.0


def initial_conformal_factor(x, density, radius):
    return 1.0


def initial_conformal_factor_gradient(x, density, radius):
    return np.zeros(len(x))


def conformal_factor_source(x, density, radius):
    return 0.0
