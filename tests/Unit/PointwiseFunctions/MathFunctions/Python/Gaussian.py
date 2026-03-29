# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def centered_coordinates(coords, center):
    return coords - center


def squared_distance_from_center(centered_coords, center):
    return np.einsum("i,i", centered_coords, centered_coords)


def call_operator(coords, amplitude, width, center):
    one_over_width = 1.0 / width
    distance = squared_distance_from_center(
        centered_coordinates(coords, center), center
    )
    return amplitude * np.exp(-1.0 * distance * np.square(one_over_width))


def first_deriv(coords, amplitude, width, center):
    one_over_width = 1.0 / width
    result = (
        -2.0
        * np.square(one_over_width)
        * call_operator(coords, amplitude, width, center)
        * centered_coordinates(coords, center)
    )
    return result


def second_deriv(coords, amplitude, width, center):
    one_over_width = 1.0 / width
    result = np.einsum(
        "i,j",
        centered_coordinates(coords, center),
        first_deriv(coords, amplitude, width, center),
    )
    result += np.eye(len(center)) * call_operator(
        coords, amplitude, width, center
    )
    return result * -2.0 * np.square(one_over_width)


def third_deriv(coords, amplitude, width, center):
    one_over_width = 1.0 / width
    centered_coords = centered_coordinates(coords, center)
    df = first_deriv(coords, amplitude, width, center)
    d2f = second_deriv(coords, amplitude, width, center)
    kronecker_delta = np.eye(len(center))
    result = np.einsum("j,ik", centered_coords, d2f)
    result += np.einsum("ij,k", kronecker_delta, df)
    result += np.einsum("jk,i", kronecker_delta, df)
    return result * -2.0 * np.square(one_over_width)
