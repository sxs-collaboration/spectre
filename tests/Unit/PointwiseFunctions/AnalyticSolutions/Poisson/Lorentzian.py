# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def field(x, constant, complex_phase):
    result = 1.0 / np.sqrt(1.0 + np.dot(x, x)) + constant
    if complex_phase != 0:
        result *= np.exp(1j * complex_phase)
    return result


def field_gradient(x, constant, complex_phase):
    dtype = float if complex_phase == 0 else complex
    result = -np.asarray(x, dtype) / np.sqrt(1.0 + np.dot(x, x)) ** 3
    if complex_phase != 0:
        result *= np.exp(1j * complex_phase)
    return result


def field_flux(x, constant, complex_phase):
    return field_gradient(x, constant, complex_phase)


def source(x, constant, complex_phase):
    result = 3.0 / np.sqrt(1.0 + np.dot(x, x)) ** 5
    if complex_phase != 0:
        result *= np.exp(1j * complex_phase)
    return result
