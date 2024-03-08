# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def deriv_normalized_normal(
    grid_frame_excision_sphere_radius,
    excision_rhat,
    normalized_normal_one_form,
    excision_normal_one_form_norm,
    inverse_spatial_metric_on_excision_boundary,
    spatial_christoffel_second_kind,
):
    Y00 = 0.5 / np.sqrt(np.pi)

    return Y00 * (
        excision_rhat / grid_frame_excision_sphere_radius
        - np.einsum(
            "k,i,j,kj->i",
            excision_rhat,
            normalized_normal_one_form,
            normalized_normal_one_form,
            inverse_spatial_metric_on_excision_boundary,
        )
        / (grid_frame_excision_sphere_radius * excision_normal_one_form_norm)
        - np.einsum(
            "i,p,j,pk,m,jkm->i",
            normalized_normal_one_form,
            normalized_normal_one_form,
            normalized_normal_one_form,
            inverse_spatial_metric_on_excision_boundary,
            excision_rhat,
            spatial_christoffel_second_kind,
        )
    )


def comoving_char_speed_derivative(
    lambda_00,
    dt_lambda_00,
    horizon_00,
    dt_horizon_00,
    grid_frame_excision_sphere_radius,
    excision_rhat,
    excision_normal_one_form,
    excision_normal_one_form_norm,
    frame_components_of_grid_shift,
    inverse_spatial_metric_on_excision_boundary,
    spatial_christoffel_second_kind,
    deriv_lapse,
    deriv_shift,
):
    Y00 = 0.5 / np.sqrt(np.pi)

    normalized_normal_one_form = (
        excision_normal_one_form / excision_normal_one_form_norm
    )

    temp = (
        excision_rhat
        * Y00
        * (
            dt_lambda_00
            - dt_horizon_00
            * (lambda_00 - grid_frame_excision_sphere_radius / Y00)
            / horizon_00
        )
        - frame_components_of_grid_shift
    )

    deriv_normal = deriv_normalized_normal(
        grid_frame_excision_sphere_radius,
        excision_rhat,
        normalized_normal_one_form,
        excision_normal_one_form_norm,
        inverse_spatial_metric_on_excision_boundary,
        spatial_christoffel_second_kind,
    )

    return np.einsum("i,i", deriv_normal, temp) + Y00 * (
        np.einsum(
            "i,j,ji", normalized_normal_one_form, excision_rhat, deriv_shift
        )
        - np.einsum("i,i", excision_rhat, deriv_lapse)
        - (dt_horizon_00 / horizon_00)
        * np.einsum("i,i", excision_rhat, normalized_normal_one_form)
    )
