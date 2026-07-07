# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def superposed_inverser_scalar_field(x, amplitude_a, amplitude_b, loc_a, loc_b):
    displacement_from_a = x - loc_a
    displacement_from_b = x - loc_b
    return amplitude_a / np.linalg.norm(
        displacement_from_a, ord=2
    ) + amplitude_b / np.linalg.norm(displacement_from_b, ord=2)


def superposed_inverser_scalar_field_derivative(
    x, amplitude_a, amplitude_b, loc_a, loc_b
):
    displacement_from_a = x - loc_a
    displacement_from_b = x - loc_b
    distance_from_a = np.linalg.norm(displacement_from_a, ord=2)
    distance_from_b = np.linalg.norm(displacement_from_b, ord=2)
    return (
        -amplitude_a * displacement_from_a / distance_from_a**3
        - amplitude_b * displacement_from_b / distance_from_b**3
    )
