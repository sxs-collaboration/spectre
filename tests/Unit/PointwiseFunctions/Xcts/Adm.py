# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def adm_mass_volume_integrand(
    conformal_factor,
    conformal_metric,
    inv_conformal_metric,
    conformal_ricci_scalar,
    extrinsic_curvature,
    trace_extrinsic_curvature,
    energy_density,
    christoffel_deriv,
):
    integrand = (1 / (16 * np.pi)) * (
        16 * np.pi * energy_density * conformal_factor**5
        + conformal_factor ** (-3)
        * np.einsum(
            "ij,kl,ik,jl",
            extrinsic_curvature,
            extrinsic_curvature,
            inv_conformal_metric,
            inv_conformal_metric,
        )
        - conformal_factor * conformal_ricci_scalar
        - conformal_factor**5 * trace_extrinsic_curvature**2
        + christoffel_deriv
    )
    return integrand


def adm_mass_surface_integrand(
    conformal_factor_deriv,
    inv_conformal_metric,
    conformal_christoffel_second_kind,
):
    contracted_christoffel = np.einsum(
        "inj,nj->i", conformal_christoffel_second_kind, inv_conformal_metric
    )
    integrand = (1 / (16 * np.pi)) * (
        contracted_christoffel - 8 * conformal_factor_deriv
    )
    return integrand
