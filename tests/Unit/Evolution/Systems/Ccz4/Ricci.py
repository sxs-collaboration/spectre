# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def deriv_christoffel_second_kind(d_conformal_christoffel_second_kind,
                                  conformal_spatial_metric,
                                  inverse_conformal_spatial_metric, field_d,
                                  field_d_up, field_p, d_field_p):
    return (
        d_conformal_christoffel_second_kind + 2.0 *
        (np.einsum("kml,ijl->kmij", field_d_up,
                   (np.einsum("jl,i", conformal_spatial_metric, field_p) +
                    np.einsum("il,j", conformal_spatial_metric, field_p) -
                    np.einsum("ij,l", conformal_spatial_metric, field_p))) -
         np.einsum("ml,ijkl->kmij", inverse_conformal_spatial_metric,
                   (np.einsum("kjl,i", field_d, field_p) +
                    np.einsum("kil,j", field_d, field_p) -
                    np.einsum("kij,l", field_d, field_p)))) -
        np.einsum("ml,ijkl->kmij", inverse_conformal_spatial_metric,
                  (np.einsum("jl,ki", conformal_spatial_metric, d_field_p) +
                   np.einsum("jl,ik", conformal_spatial_metric, d_field_p) +
                   np.einsum("il,kj", conformal_spatial_metric, d_field_p) +
                   np.einsum("il,jk", conformal_spatial_metric, d_field_p) -
                   np.einsum("ij,kl", conformal_spatial_metric, d_field_p) -
                   np.einsum("ij,lk", conformal_spatial_metric, d_field_p))) /
        2.0)


def spatial_ricci_tensor(christoffel_second_kind,
                         d_conformal_christoffel_second_kind,
                         conformal_spatial_metric,
                         inverse_conformal_spatial_metric, field_d, field_d_up,
                         field_p, d_field_p):
    d_christoffel_second_kind = deriv_christoffel_second_kind(
        d_conformal_christoffel_second_kind, conformal_spatial_metric,
        inverse_conformal_spatial_metric, field_d, field_d_up, field_p,
        d_field_p)

    return (
        np.einsum("mmij", d_christoffel_second_kind) -
        np.einsum("jmim", d_christoffel_second_kind) + np.einsum(
            "lij,mlm", christoffel_second_kind, christoffel_second_kind) -
        np.einsum("lim,mlj", christoffel_second_kind, christoffel_second_kind))
