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


# Functions for testing ForceFreeConstraints
def tilde_e_or_tilde_b_squared(tilde_e_or_tilde_b, spatial_metric):
    one_form = np.einsum("a, ia", tilde_e_or_tilde_b, spatial_metric)
    return np.einsum("a, a", one_form, tilde_e_or_tilde_b)


def tilde_e_dot_tilde_b_compute(tilde_e, tilde_b, spatial_metric):
    magnetic_field_one_form = np.einsum("a, ia", tilde_b, spatial_metric)
    return np.einsum("a, a", magnetic_field_one_form, tilde_e)


def e_dot_b_compute(tilde_e, tilde_b, sqrt_det_spatial_metric, spatial_metric):
    magnetic_field_one_form = np.einsum("a, ia", tilde_b, spatial_metric)
    return (
        np.einsum("a, a", magnetic_field_one_form, tilde_e)
        / sqrt_det_spatial_metric**2
    )


def magnetic_dominance_violation_compute(
    tilde_e_squared, tilde_b_squared, sqrt_det_spatial_metric
):
    return (
        max(tilde_e_squared - tilde_b_squared, 0.0)
        / sqrt_det_spatial_metric**2
    )


# end functions for testing ForceFreeConstraints
