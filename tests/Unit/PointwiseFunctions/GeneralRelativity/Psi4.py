# Distributed under the MIT License.
# See LICENSE.txt for details.

import cmath
import math

import numpy as np

from .ProjectionOperators import transverse_projection_operator
from .WeylPropagating import weyl_propagating_modes


def psi_4(
    spatial_ricci,
    extrinsic_curvature,
    cov_deriv_extrinsic_curvature,
    spatial_metric,
    inv_spatial_metric,
    inertial_coords,
):
    magnitude_inertial = math.sqrt(
        np.einsum("a,b,ab", inertial_coords, inertial_coords, spatial_metric)
    )
    if magnitude_inertial != 0.0:
        r_hat = np.einsum("a", inertial_coords / magnitude_inertial)
    else:
        r_hat = np.einsum("a", inertial_coords * 0.0)

    lower_r_hat = np.einsum("a,ab", r_hat, spatial_metric)

    inv_projection_tensor = transverse_projection_operator(
        inv_spatial_metric, r_hat
    )
    projection_tensor = transverse_projection_operator(
        spatial_metric, lower_r_hat
    )
    projection_up_lo = np.einsum("ab,ac", inv_projection_tensor, spatial_metric)

    u8_plus = weyl_propagating_modes(
        spatial_ricci,
        extrinsic_curvature,
        inv_spatial_metric,
        cov_deriv_extrinsic_curvature,
        r_hat,
        inv_projection_tensor,
        projection_tensor,
        projection_up_lo,
        1,
    )

    x_coord = np.zeros((3))
    x_coord[0] = 1
    x_component = np.einsum("a,b,ab", x_coord, r_hat, spatial_metric)
    x_hat = x_coord - (x_component * r_hat)
    magnitude_x = math.sqrt(np.einsum("a,b,ab", x_hat, x_hat, spatial_metric))
    if magnitude_x != 0.0:
        x_hat = np.einsum("a", x_hat / magnitude_x)
    else:
        x_hat = np.einsum("a", x_hat * 0.0)
    y_coord = np.zeros((3))
    y_coord[1] = 1
    y_component = np.einsum("a,b,ab", y_coord, r_hat, spatial_metric)
    y_hat = y_coord - (y_component * r_hat)
    y_component = np.einsum("a,b,ab", y_coord, x_hat, spatial_metric)
    y_hat = y_hat - (y_component * x_hat)
    magnitude_y = math.sqrt(np.einsum("a,b,ab", y_hat, y_hat, spatial_metric))
    if magnitude_y != 0.0:
        y_hat = np.einsum("a", y_hat / magnitude_y)
    else:
        y_hat = np.einsum("a", y_hat * 0.0)
    m_bar = x_hat - (y_hat * complex(0.0, 1.0))

    return -0.5 * np.einsum("ab,a,b", u8_plus, m_bar, m_bar)
