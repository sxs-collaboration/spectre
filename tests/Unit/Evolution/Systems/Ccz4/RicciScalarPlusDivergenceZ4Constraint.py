# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def ricci_scalar_plus_divergence_z4_constraint(
    conformal_factor_squared, inverse_conformal_spatial_metric,
    spatial_ricci_tensor, grad_spatial_z4_constraint):
    return conformal_factor_squared * np.einsum(
        "ij,ij", inverse_conformal_spatial_metric,
        spatial_ricci_tensor + grad_spatial_z4_constraint +
        np.einsum("ji", grad_spatial_z4_constraint))
