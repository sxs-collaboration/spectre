# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def spatial_z4_constraint(conformal_spatial_metric,
                          gamma_hat_minus_contracted_conformal_christoffel):
    return (0.5 * np.einsum("ij,j", conformal_spatial_metric,
                            gamma_hat_minus_contracted_conformal_christoffel))


def upper_spatial_z4_constraint(
    conformal_factor_squared,
    gamma_hat_minus_contracted_conformal_christoffel):
    return (0.5 * conformal_factor_squared *
            gamma_hat_minus_contracted_conformal_christoffel)
