# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from GaussianPlusConstant import (
    call_operator as GaussianPlusConstant_call_operator,
)


def call_operator(
    coords,
    time,
    constant,
    amplitude_1,
    width_1,
    unused_center_1,
    amplitude_2,
    width_2,
    unused_center_2,
    amplitude_3,
    width_3,
    center_3,
):
    center_1 = np.asarray([16.0 + time * -1.0e-3, 0.0, 0.0])
    center_2 = np.asarray([-16.0 + time * 2.0e-3, 0.0, 0.0])
    return (
        GaussianPlusConstant_call_operator(
            coords,
            time,
            constant,
            amplitude_1,
            width_1,
            center_1,
        )
        + GaussianPlusConstant_call_operator(
            coords,
            time,
            0.0,
            amplitude_2,
            width_2,
            center_2,
        )
        + GaussianPlusConstant_call_operator(
            coords,
            time,
            0.0,
            amplitude_3,
            width_3,
            center_3,
        )
    )
