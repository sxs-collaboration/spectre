# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions testing SpacetimeDerivativeOfGothG


def spacetime_deriv_of_goth_g(
    inverse_spacetime_metric,
    da_spacetime_metric,
    lapse,
    da_lapse,
    sqrt_det_spatial_metric,
    da_det_spatial_metric,
):
    return np.tensordot(
        da_lapse * sqrt_det_spatial_metric
        + 0.5 * lapse * da_det_spatial_metric / sqrt_det_spatial_metric,
        inverse_spacetime_metric,
        axes=0,
    ) - lapse * sqrt_det_spatial_metric * np.einsum(
        "im,jn,hmn",
        inverse_spacetime_metric,
        inverse_spacetime_metric,
        da_spacetime_metric,
    )


# End functions for testing SpacetimeDerivativeOfGothG
