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
    distance_squared = squared_distance_from_center(
        centered_coordinates(coords, center), center)
    return amplitude * np.exp(
        -1.0 * distance_squared * np.square(one_over_width)) + constant


def function_of_time(time):
    # The test in tests/Unit/Helpers/Evolution/Systems/GeneralizedHarmonic/ \
    # ConstraintDamping/TestHelpers.hpp hard-codes the following
    # FunctionOfTime for evaluating time-dependent DampingFunctions
    a = [1.0, 0.2, 0.03, 0.004]
    return a[0] + a[1] * (time + 1.0) + a[2] * np.square(time + 1.0) + a[3] * (
        time + 1.0) * (time + 1.0) * (time + 1.0)


def time_dependent_triple_gaussian_call_operator(
    coords, time, constant, amplitude_1, width_1, center_1, amplitude_2,
    width_2, center_2, amplitude_3, width_3, center_3):
    factor_scaling_widths = 1.0 / function_of_time(time)
    return gaussian_plus_constant_call_operator(
        coords, time, constant, amplitude_1, width_1 * factor_scaling_widths,
        center_1) + gaussian_plus_constant_call_operator(
            coords, time, 0.0, amplitude_2, width_2 * factor_scaling_widths,
            center_2) + gaussian_plus_constant_call_operator(
                coords, time, 0.0, amplitude_3,
                width_3 * factor_scaling_widths, center_3)
