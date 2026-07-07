# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def inverser_scalar_field(x, amplitude):
    return amplitude / np.linalg.norm(x, ord=2)


def inverser_scalar_field_derivative(x, amplitude):
    distance_from_center = np.linalg.norm(x, ord=2)
    return -amplitude * x / distance_from_center**3
