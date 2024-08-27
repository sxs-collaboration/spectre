# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def adm_linear_momentum_surface_integrand(
    conformal_factor,
    inv_spatial_metric,
    inv_extrinsic_curvature,
    trace_extrinsic_curvature,
):
    return (
        1.0
        / (8.0 * np.pi)
        * conformal_factor**10
        * (
            inv_extrinsic_curvature
            - trace_extrinsic_curvature * inv_spatial_metric
        )
    )


def adm_linear_momentum_volume_integrand(
    surface_integrand,
    conformal_factor,
    conformal_factor_deriv,
    conformal_metric,
    inv_conformal_metric,
    conformal_christoffel_second_kind,
    conformal_christoffel_contracted,
):
    first_term = np.einsum(
        "ijk,jk->i",
        conformal_christoffel_second_kind,
        surface_integrand,
    )

    second_term = np.einsum(
        "k,ik->i",
        conformal_christoffel_contracted,
        surface_integrand,
    )

    third_term = 2.0 * np.einsum(
        "jk,jk,il,l->i",
        conformal_metric,
        surface_integrand,
        inv_conformal_metric,
        conformal_factor_deriv / conformal_factor,
    )

    return -(first_term + second_term - third_term)
