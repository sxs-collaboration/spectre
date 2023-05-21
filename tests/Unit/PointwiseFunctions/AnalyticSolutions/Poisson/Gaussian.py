# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def field(x, amplitude, width, center):
    return amplitude * np.exp(-np.linalg.norm(x - center) ** 2 / width**2)


def field_gradient(x, amplitude, width, center):
    return -2.0 * x / width**2 * field(x, amplitude, width, center)


def field_flux(x, amplitude, width, center):
    return field_gradient(x, amplitude, width, center)


def source(x, amplitude, width, center):
    return (
        2.0 * len(x) / width**2 - 4.0 * np.linalg.norm(x) ** 2 / width**4
    ) * field(x, amplitude, width, center)
