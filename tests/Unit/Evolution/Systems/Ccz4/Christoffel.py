# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                      field_d):
    return np.einsum("kl,ijl->kij", inverse_conformal_spatial_metric,
                     (np.einsum("ijl", field_d) + np.einsum("jil", field_d) -
                      np.einsum("lij", field_d)))


def christoffel_second_kind(conformal_spatial_metric,
                            inverse_conformal_spatial_metric, field_p,
                            conformal_christoffel_second_kind):
    return (
        np.einsum("kij->kij", conformal_christoffel_second_kind) -
        (np.einsum("kl,ijl->kij", inverse_conformal_spatial_metric,
                   (np.einsum("jl,i", conformal_spatial_metric, field_p) +
                    np.einsum("il,j", conformal_spatial_metric, field_p) -
                    np.einsum("ij,l", conformal_spatial_metric, field_p)))))
