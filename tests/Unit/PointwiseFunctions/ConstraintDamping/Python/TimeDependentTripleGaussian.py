# Distributed under the MIT License.
# See LICENSE.txt for details.

from ConstraintDampingHelpers import function_of_time
from GaussianPlusConstant import (
    call_operator as GaussianPlusConstant_call_operator,
)


def call_operator(
    coords,
    time,
    constant,
    amplitude_1,
    width_1,
    center_1,
    amplitude_2,
    width_2,
    center_2,
    amplitude_3,
    width_3,
    center_3,
):
    factor_scaling_widths = 1.0 / function_of_time(time)
    return (
        GaussianPlusConstant_call_operator(
            coords,
            time,
            constant,
            amplitude_1,
            width_1 * factor_scaling_widths,
            center_1,
        )
        + GaussianPlusConstant_call_operator(
            coords,
            time,
            0.0,
            amplitude_2,
            width_2 * factor_scaling_widths,
            center_2,
        )
        + GaussianPlusConstant_call_operator(
            coords,
            time,
            0.0,
            amplitude_3,
            width_3 * factor_scaling_widths,
            center_3,
        )
    )
