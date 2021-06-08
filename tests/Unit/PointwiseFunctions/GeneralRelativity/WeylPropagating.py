# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from .WeylElectric import weyl_electric_tensor


def weyl_propagating_modes(ricci, extrinsic_curvature, inverse_spatial_metric,
                           cov_deriv_ex_curv, unit_normal_vector,
                           projection_IJ, projection_ij, projection_Ij, sign):
    tmp = weyl_electric_tensor(ricci, extrinsic_curvature,
                               inverse_spatial_metric)
    tmp = tmp - sign * np.einsum('k,kij->ij', unit_normal_vector,
                                 cov_deriv_ex_curv)
    tmp = tmp + sign * 0.5 * np.einsum('k,jik->ij', unit_normal_vector,
                                       cov_deriv_ex_curv)
    tmp = tmp + sign * 0.5 * np.einsum('k,ijk->ij', unit_normal_vector,
                                       cov_deriv_ex_curv)

    return np.einsum('ki,lj,kl->ij', projection_Ij,
                     projection_Ij, tmp) - 0.5 * np.einsum(
                         'kl,ij,kl->ij', projection_IJ, projection_ij, tmp)


def weyl_propagating_mode_plus(ricci, extrinsic_curvature,
                               inverse_spatial_metric, cov_deriv_ex_curv,
                               unit_normal_vector, projection_IJ,
                               projection_ij, projection_Ij):
    return weyl_propagating_modes(ricci, extrinsic_curvature,
                                  inverse_spatial_metric, cov_deriv_ex_curv,
                                  unit_normal_vector, projection_IJ,
                                  projection_ij, projection_Ij, 1)


def weyl_propagating_mode_minus(ricci, extrinsic_curvature,
                                inverse_spatial_metric, cov_deriv_ex_curv,
                                unit_normal_vector, projection_IJ,
                                projection_ij, projection_Ij):
    return weyl_propagating_modes(ricci, extrinsic_curvature,
                                  inverse_spatial_metric, cov_deriv_ex_curv,
                                  unit_normal_vector, projection_IJ,
                                  projection_ij, projection_Ij, -1)
