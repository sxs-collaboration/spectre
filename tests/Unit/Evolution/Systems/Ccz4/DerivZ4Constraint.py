# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def grad_spatial_z4_constraint(
    spatial_z4_constraint, conformal_spatial_metric, christoffel_second_kind,
    field_d, gamma_hat_minus_contracted_conformal_christoffel,
    d_gamma_hat_minus_contracted_conformal_christoffel):
    return (
        np.einsum("ijl,l", field_d,
                  gamma_hat_minus_contracted_conformal_christoffel) +
        0.5 * np.einsum("jl,il", conformal_spatial_metric,
                        d_gamma_hat_minus_contracted_conformal_christoffel) -
        np.einsum("lij,l", christoffel_second_kind, spatial_z4_constraint))
