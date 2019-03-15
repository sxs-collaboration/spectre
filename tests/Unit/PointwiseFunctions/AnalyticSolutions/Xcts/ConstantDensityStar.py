# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from scipy.optimize import newton


def compute_alpha(density, radius):
    def f(a):
        return density * radius**2 - 3. / (2. * np.pi) * a**10 / (1. + a**2)**6

    def fprime(a):
        return 3. * a**9 * (a**2 - 5.) / (1. + a**2)**7 / np.pi

    return newton(func=f, fprime=fprime, x0=2. * np.sqrt(5.))


def sobolov(r, alpha, radius):
    return np.sqrt(alpha * radius / (r**2 + (alpha * radius)**2))


def conformal_factor(x, density, radius):
    alpha = compute_alpha(density, radius)
    C = (3. / (2. * np.pi * density))**(1./4.)
    r = np.linalg.norm(x)
    if r <= radius:
        return C * sobolov(r, alpha, radius)
    else:
        beta = radius * (C * sobolov(radius, alpha, radius) - 1.)
        return beta / r + 1.


def initial_conformal_factor(x, density, radius):
    return 1.


def initial_conformal_factor_gradient(x, density, radius):
    return np.zeros(len(x))


def conformal_factor_source(x, density, radius):
    return 0.


def conformal_factor_gradient_source(x, density, radius):
    return np.zeros(len(x))
