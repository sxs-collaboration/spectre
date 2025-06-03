# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def geodesic_equation(
    x,
    pi,
    lnp0,
    lapse,
    deriv_lapse,
    shift,
    deriv_shift,
    inv_spatial_metric,
    deriv_inv_spatial_metric,
    extrinsic_curvature,
):
    pi_upper = np.einsum("ij,j", inv_spatial_metric, pi)
    dt_lnp0 = -np.einsum("i,i", deriv_lapse, pi_upper) + lapse * np.einsum(
        "ij,i,j", extrinsic_curvature, pi_upper, pi_upper
    )
    dt_x = lapse * pi_upper - shift
    dt_pi = (
        -deriv_lapse
        - dt_lnp0 * pi
        + np.einsum("ik,k", deriv_shift, pi)
        - 0.5 * lapse * np.einsum("ijk,j,k", deriv_inv_spatial_metric, pi, pi)
    )
    return [dt_x, dt_pi, dt_lnp0]


def dt_x(*args):
    return geodesic_equation(*args)[0]


def dt_pi(*args):
    return geodesic_equation(*args)[1]


def dt_lnp0(*args):
    return geodesic_equation(*args)[2]
