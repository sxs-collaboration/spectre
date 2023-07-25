# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing ElectromagneticVariables
def electric_field_compute(tilde_e, sqrt_det_spatial_metric):
    return tilde_e / sqrt_det_spatial_metric


def magnetic_field_compute(tilde_b, sqrt_det_spatial_metric):
    return tilde_b / sqrt_det_spatial_metric


def charge_density_compute(tilde_q, sqrt_det_spatial_metric):
    return tilde_q / sqrt_det_spatial_metric


def electric_current_density_compute(tilde_j, lapse, sqrt_det_spatial_metric):
    return tilde_j / (lapse * sqrt_det_spatial_metric)


# end functions for testing ElectromagneticVariables


# Functions for testing MaskNeutronStarInterior
def compute_ns_interior_mask(coords):
    r_squared = np.einsum("a, a", coords, coords)

    if r_squared < 1.0:
        return -1.0
    else:
        return 1.0


# end functions for testing MaskNeutronStarInterior
