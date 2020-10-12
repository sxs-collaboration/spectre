# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data_u(u, flux_u, normal_covector, mesh_velocity,
                      normal_dot_mesh_velocity):
    return u


def dg_package_data_normal_dot_flux(u, flux_u, normal_covector, mesh_velocity,
                                    normal_dot_mesh_velocity):
    return np.einsum("i,i", normal_covector, flux_u)


def dg_package_data_char_speed(u, flux_u, normal_covector, mesh_velocity,
                               normal_dot_mesh_velocity):
    result = u if normal_covector[0] > 0.0 else -u
    if normal_dot_mesh_velocity is None:
        return result
    else:
        return result - normal_dot_mesh_velocity


def dg_boundary_terms_u(interior_u, interior_normal_dot_flux_u,
                        interior_char_speed, exterior_u,
                        exterior_normal_dot_flux_u, exterior_char_speed,
                        use_strong_form):
    lambda_min = np.minimum(np.minimum(0.0, interior_char_speed),
                            -exterior_char_speed)
    lambda_max = np.maximum(np.maximum(0.0, interior_char_speed),
                            -exterior_char_speed)
    result = (
        (lambda_max * interior_normal_dot_flux_u +
         lambda_min * exterior_normal_dot_flux_u) + lambda_max * lambda_min *
        (exterior_u - interior_u)) / (lambda_max - lambda_min)

    if use_strong_form:
        result -= interior_normal_dot_flux_u
    return result
