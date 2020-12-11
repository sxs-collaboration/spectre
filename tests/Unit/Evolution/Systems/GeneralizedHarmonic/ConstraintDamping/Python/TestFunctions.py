# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def centered_coordinates(coords, center):
    return coords - center


def squared_distance_from_center(centered_coords, center):
    return np.einsum("i,i", centered_coords, centered_coords)


def gaussian_plus_constant_call_operator(coords, time, constant, amplitude,
                                         width, center):
    one_over_width = 1.0 / width
    distance = squared_distance_from_center(
        centered_coordinates(coords, center), center)
    return amplitude * np.exp(
        -1.0 * distance * np.square(one_over_width)) + constant
