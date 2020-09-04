# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def centered_coordinates(coords, center):
    return coords - center


def squared_distance_from_center(centered_coords, center):
    return np.einsum("i,i", centered_coords, centered_coords)


def gaussian_call_operator(coords, amplitude, width, center):
    one_over_width = 1.0 / width
    distance = squared_distance_from_center(
        centered_coordinates(coords, center), center)
    return amplitude * np.exp(-1.0 * distance * np.square(one_over_width))


def gaussian_first_deriv(coords, amplitude, width, center):
    one_over_width = 1.0 / width
    result = -2.0 * np.square(one_over_width) * gaussian_call_operator(
        coords, amplitude, width, center) * centered_coordinates(
            coords, center)
    return result


def gaussian_second_deriv(coords, amplitude, width, center):
    one_over_width = 1.0 / width
    result = np.einsum("i,j", centered_coordinates(coords, center),
                       gaussian_first_deriv(coords, amplitude, width, center))
    result += np.eye(len(center)) * gaussian_call_operator(
        coords, amplitude, width, center)
    return result * -2.0 * np.square(one_over_width)


def gaussian_third_deriv(coords, amplitude, width, center):
    one_over_width = 1.0 / width
    centered_coords = centered_coordinates(coords, center)
    df = gaussian_first_deriv(coords, amplitude, width, center)
    d2f = gaussian_second_deriv(coords, amplitude, width, center)
    kronecker_delta = np.eye(len(center))
    result = np.einsum("j,ik", centered_coords, d2f)
    result += np.einsum("ij,k", kronecker_delta, df)
    result += np.einsum("jk,i", kronecker_delta, df)
    return result * -2.0 * np.square(one_over_width)


def sinusoid_call_operator(coords, amplitude, wavenumber, phase):
    return amplitude * np.sin(wavenumber * coords + phase)[0]


def sinusoid_first_deriv(coords, amplitude, wavenumber, phase):
    return amplitude * wavenumber * np.cos(wavenumber * coords + phase)


def sinusoid_second_deriv(coords, amplitude, wavenumber, phase):
    return np.array([
        -amplitude * np.square(wavenumber) *
        np.sin(wavenumber * coords + phase)
    ])


def sinusoid_third_deriv(coords, amplitude, wavenumber, phase):
    return np.array([[
        -amplitude * wavenumber * np.square(wavenumber) *
        np.cos(wavenumber * coords + phase)
    ]])


def pow_x_call_operator(coords, power):
    return np.power(coords, power)[0]


def pow_x_first_deriv(coords, power):
    if power == 0.0:
        return np.array([0.0])
    else:
        return power * np.power(coords, power - 1.0)


def pow_x_second_deriv(coords, power):
    if power == 0.0 or power == 1.0:
        return np.array([[0.0]])
    else:
        return np.array([(power - 1.0) * power * np.power(coords, power - 2.0)
                         ])


def pow_x_third_deriv(coords, power):
    if power == 0.0 or power == 1.0 or power == 2.0:
        return np.array([[[0.0]]])
    else:
        return np.array([[(power - 2.0) * (power - 1.0) * power *
                          np.power(coords, power - 3.0)]])
