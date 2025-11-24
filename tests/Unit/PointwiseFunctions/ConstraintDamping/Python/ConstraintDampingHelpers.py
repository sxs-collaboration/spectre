# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def centered_coordinates(coords, center):
    return coords - center


def squared_distance_from_center(centered_coords, center):
    return np.einsum("i,i", centered_coords, centered_coords)


def function_of_time(time):
    a = [1.0, 0.2, 0.03, 0.004]
    return (
        a[0]
        + a[1] * (time + 1.0)
        + a[2] * np.square(time + 1.0)
        + a[3] * (time + 1.0) * (time + 1.0) * (time + 1.0)
    )
