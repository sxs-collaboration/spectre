# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def exponential_coupling(psi, lambda_coeff, gamma_coeff):
    return lambda_coeff * np.exp(-gamma_coeff * psi)


def exponential_coupling_prime(psi, lambda_coeff, gamma_coeff):
    return -gamma_coeff * lambda_coeff * np.exp(-gamma_coeff * psi)


def exponential_coupling_prime_prime(psi, lambda_coeff, gamma_coeff):
    return gamma_coeff**2 * lambda_coeff * np.exp(-gamma_coeff * psi)


def quarticpolynomial_coupling(psi, linear, quadratic, cubic, quartic):
    return (
        linear * psi
        + quadratic * psi**2
        + cubic * psi**3
        + quartic * psi**4
    )


def quarticpolynomial_coupling_prime(psi, linear, quadratic, cubic, quartic):
    return (
        linear
        + 2.0 * quadratic * psi
        + 3.0 * cubic * psi**2
        + 4.0 * quartic * psi**3
    )


def quarticpolynomial_coupling_prime_prime(
    psi, linear, quadratic, cubic, quartic
):
    return 2.0 * quadratic + 6.0 * cubic * psi + 12.0 * quartic * psi**2
