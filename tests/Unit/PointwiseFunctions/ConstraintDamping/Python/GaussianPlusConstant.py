# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from ConstraintDampingHelpers import (
    centered_coordinates,
    squared_distance_from_center,
)


def call_operator(coords, time, constant, amplitude, width, center):
    one_over_width = 1.0 / width
    distance_squared = squared_distance_from_center(
        centered_coordinates(coords, center), center
    )
    return (
        amplitude * np.exp(-1.0 * distance_squared * np.square(one_over_width))
        + constant
    )
