# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from PointwiseFunctions.GeneralRelativity.ComputeSpacetimeQuantities import (
    extrinsic_curvature)


def spatial_metric(conformal_factor, conformal_metric):
    return conformal_factor**4 * conformal_metric


def inv_spatial_metric(conformal_factor, inv_conformal_metric):
    return conformal_factor**(-4) * inv_conformal_metric


def lapse(conformal_factor, lapse_times_conformal_factor):
    return lapse_times_conformal_factor / conformal_factor


def shift(shift_excess, shift_background):
    return shift_excess + shift_background


def hamiltonian_constraint(spatial_ricci_tensor, extrinsic_curvature,
                           local_inv_spatial_metric):
    return (
        np.einsum('ij,ij', local_inv_spatial_metric, spatial_ricci_tensor) +
        np.einsum('ij,ij', local_inv_spatial_metric, extrinsic_curvature)**2 -
        np.einsum('ij,kl,ik,jl', local_inv_spatial_metric,
                  local_inv_spatial_metric, extrinsic_curvature,
                  extrinsic_curvature))


def momentum_constraint(cov_deriv_extrinsic_curvature,
                        local_inv_spatial_metric):
    return (np.einsum('jk,jki', local_inv_spatial_metric,
                      cov_deriv_extrinsic_curvature) -
            np.einsum('jk,ijk', local_inv_spatial_metric,
                      cov_deriv_extrinsic_curvature))
