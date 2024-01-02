# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    u,
    flux_u,
    velocity_field,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    normal_dot_velocity = np.einsum("i,i", velocity_field, normal_covector)
    return (
        u,
        np.asarray(np.einsum("i,i", normal_covector, flux_u)),
        np.asarray(
            np.abs(normal_dot_velocity)
            if normal_dot_mesh_velocity is None
            else np.abs(normal_dot_velocity - normal_dot_mesh_velocity)
        ),
    )


def dg_boundary_terms(
    interior_u,
    interior_normal_dot_flux_u,
    interior_abs_char_speed,
    exterior_u,
    exterior_normal_dot_flux_u,
    exterior_abs_char_speed,
    use_strong_form,
):
    if use_strong_form:
        return (
            np.asarray(
                -0.5 * (interior_normal_dot_flux_u + exterior_normal_dot_flux_u)
                - 0.5
                * np.maximum(interior_abs_char_speed, exterior_abs_char_speed)
                * (exterior_u - interior_u)
            ),
        )
    else:
        return (
            np.asarray(
                0.5 * (interior_normal_dot_flux_u - exterior_normal_dot_flux_u)
                - 0.5
                * np.maximum(interior_abs_char_speed, exterior_abs_char_speed)
                * (exterior_u - interior_u)
            ),
        )
