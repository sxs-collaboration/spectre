# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def nonic_ramp_function(time, t_start, t_ramp):
    if time < t_start:
        ramp_factor = 0.0
    elif time < t_ramp + t_start:
        t_star = (time - t_start) / t_ramp
        ramp_factor = np.power(t_star, 5) * (
            126
            + t_star * (-420 + t_star * (540 + t_star * (-315 + 70 * t_star)))
        )
    else:
        ramp_factor = 1.0
    return ramp_factor
