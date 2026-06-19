# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def hamiltonian_constraint_in_vacuum(
    ricci_scalar,
    trace_extrinsic_curvature,
    inverse_spatial_metric,
    extrinsic_curvature,
):
    extrinsic_curvature_square = np.einsum(
        "ij,ik,jl,kl->",
        extrinsic_curvature,
        inverse_spatial_metric,
        inverse_spatial_metric,
        extrinsic_curvature,
    )
    return (
        ricci_scalar
        + trace_extrinsic_curvature**2
        - extrinsic_curvature_square
    )
