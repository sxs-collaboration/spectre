# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def photon_geodesic_equation_with_constraint(
    x,
    pi,
    lapse,
    deriv_lapse,
    shift,
    deriv_shift,
    inv_spatial_metric,
    deriv_spatial_metric,
    extrinsic_curvature,
):
    # 1. Contravariant spatial momentum: pi^i = gamma^{ij} pi_j
    upper_pi = np.einsum("ij,j", inv_spatial_metric, pi)

    # 2. Null constraint: p^0 = sqrt(gamma^{ij} pi_i pi_j) / alpha
    pi_squared = np.einsum("i,i", upper_pi, pi)
    p0 = np.sqrt(pi_squared) / lapse

    # 3. dt_lnp0 = 0  (constraint-enforced, not evolved)
    dt_lnp0 = 0.0 * lapse  # zero, same shape as lapse

    # 4. dx^i/dt = pi^i / p^0 - beta^i
    dt_x = upper_pi / p0 - shift

    # 5. dp_i/dt = -alpha * (d_i alpha) * p^0
    #              + pi_k * d_i beta^k
    #              + (1 / (2 p^0)) * pi^m pi^n * d_i gamma_{mn}
    dt_pi = (
        -lapse * deriv_lapse * p0
        + np.einsum("ik,k", deriv_shift, pi)
        + (0.5 / p0)
        * np.einsum("imn,m,n", deriv_spatial_metric, upper_pi, upper_pi)
    )

    return [dt_x, dt_pi, p0, dt_lnp0]


def dt_x(*args):
    return photon_geodesic_equation_with_constraint(*args)[0]


def dt_pi(*args):
    return photon_geodesic_equation_with_constraint(*args)[1]


def current_p0(*args):
    return photon_geodesic_equation_with_constraint(*args)[2]


def current_dt_lnp0(*args):
    return photon_geodesic_equation_with_constraint(*args)[3]
