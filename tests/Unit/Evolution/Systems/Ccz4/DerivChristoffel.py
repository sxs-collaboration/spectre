# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def deriv_conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                            field_d, d_field_d, field_d_up):
    return (
        -2.0 * np.einsum("kml,ijl->kmij", field_d_up,
                         (np.einsum("ijl", field_d) + np.einsum(
                             "jil", field_d) - np.einsum("lij", field_d))) +
        np.einsum(
            "ml,ijkl->kmij", inverse_conformal_spatial_metric,
            (np.einsum("kijl", d_field_d) + np.einsum("ikjl", d_field_d) +
             np.einsum("kjil", d_field_d) + np.einsum("jkil", d_field_d) -
             np.einsum("klij", d_field_d) - np.einsum("lkij", d_field_d))) /
        2.0)


def deriv_contracted_conformal_christoffel_second_kind(
    inverse_conformal_spatial_metric, field_d_up,
    conformal_christoffel_second_kind, d_conformal_christoffel_second_kind):
    return (-2.0 * np.einsum("kjl,ijl->ki", field_d_up,
                             conformal_christoffel_second_kind) +
            np.einsum("jl,kijl->ki", inverse_conformal_spatial_metric,
                      d_conformal_christoffel_second_kind))
