# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from PointwiseFunctions.Xcts.ExtrinsicCurvature import extrinsic_curvature


def spatial_metric(conformal_factor, conformal_metric):
    return conformal_factor**4 * conformal_metric


def inv_spatial_metric(conformal_factor, inv_conformal_metric):
    return conformal_factor ** (-4) * inv_conformal_metric


def spatial_christoffel_second_kind(
    conformal_factor,
    deriv_conformal_factor,
    conformal_metric,
    inv_conformal_metric,
    conformal_christoffel_second_kind,
):
    # Eq. (3.7) in Baumgarte/Shapiro
    krond = np.eye(3)
    return conformal_christoffel_second_kind + 2.0 * (
        np.einsum("ij,k->ijk", krond, deriv_conformal_factor / conformal_factor)
        + np.einsum(
            "ik,j->ijk", krond, deriv_conformal_factor / conformal_factor
        )
        - np.einsum(
            "jk,il,l->ijk",
            conformal_metric,
            inv_conformal_metric,
            deriv_conformal_factor / conformal_factor,
        )
    )


def lapse(conformal_factor, lapse_times_conformal_factor):
    return lapse_times_conformal_factor / conformal_factor


def shift(shift_excess, shift_background):
    return shift_excess + shift_background


def hamiltonian_constraint(
    spatial_ricci_tensor, extrinsic_curvature, local_inv_spatial_metric
):
    return (
        np.einsum("ij,ij", local_inv_spatial_metric, spatial_ricci_tensor)
        + np.einsum("ij,ij", local_inv_spatial_metric, extrinsic_curvature) ** 2
        - np.einsum(
            "ij,kl,ik,jl",
            local_inv_spatial_metric,
            local_inv_spatial_metric,
            extrinsic_curvature,
            extrinsic_curvature,
        )
    )


def momentum_constraint(
    cov_deriv_extrinsic_curvature, local_inv_spatial_metric
):
    return np.einsum(
        "jk,jki", local_inv_spatial_metric, cov_deriv_extrinsic_curvature
    ) - np.einsum(
        "jk,ijk", local_inv_spatial_metric, cov_deriv_extrinsic_curvature
    )
