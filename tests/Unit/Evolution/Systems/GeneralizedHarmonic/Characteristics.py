# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Test functions for characteristic speeds
def char_speed_upsi(gamma1, lapse, shift, unit_normal):
    return -(1.0 + gamma1) * np.dot(shift, unit_normal)


def char_speed_upsi_moving_mesh(
    gamma1, lapse, shift, unit_normal, mesh_velocity
):
    return -(1.0 + gamma1) * np.dot(shift, unit_normal) - (
        1.0 + gamma1
    ) * np.dot(mesh_velocity, unit_normal)


def char_speed_uzero(gamma1, lapse, shift, unit_normal):
    return -np.dot(shift, unit_normal)


def char_speed_uzero_moving_mesh(
    gamma1, lapse, shift, unit_normal, mesh_velocity
):
    return -np.dot(shift, unit_normal) - np.dot(mesh_velocity, unit_normal)


def char_speed_uplus(gamma1, lapse, shift, unit_normal):
    return -np.dot(shift, unit_normal) + lapse


def char_speed_uplus_moving_mesh(
    gamma1, lapse, shift, unit_normal, mesh_velocity
):
    return (
        -np.dot(shift, unit_normal) + lapse - np.dot(mesh_velocity, unit_normal)
    )


def char_speed_uminus(gamma1, lapse, shift, unit_normal):
    return -np.dot(shift, unit_normal) - lapse


def char_speed_uminus_moving_mesh(
    gamma1, lapse, shift, unit_normal, mesh_velocity
):
    return (
        -np.dot(shift, unit_normal) - lapse - np.dot(mesh_velocity, unit_normal)
    )


# End test functions for characteristic speeds


# Test functions for characteristic fields
def char_field_upsi(
    gamma2, inverse_spatial_metric, spacetime_metric, pi, phi, normal_one_form
):
    return spacetime_metric


def char_field_uzero(
    gamma2, inverse_spatial_metric, spacetime_metric, pi, phi, normal_one_form
):
    normal_vector = np.einsum("ij,j", inverse_spatial_metric, normal_one_form)
    projection_tensor = np.identity(len(normal_vector)) - np.einsum(
        "i,j", normal_one_form, normal_vector
    )
    return np.einsum("ij,jab->iab", projection_tensor, phi)


def char_field_uplus(
    gamma2, inverse_spatial_metric, spacetime_metric, pi, phi, normal_one_form
):
    normal_vector = np.einsum("ij,j", inverse_spatial_metric, normal_one_form)
    phi_dot_normal = np.einsum("i,iab->ab", normal_vector, phi)
    return pi + 1 * phi_dot_normal - (gamma2 * spacetime_metric)


def char_field_uminus(
    gamma2, inverse_spatial_metric, spacetime_metric, pi, phi, normal_one_form
):
    normal_vector = np.einsum("ij,j", inverse_spatial_metric, normal_one_form)
    phi_dot_normal = np.einsum("i,iab->ab", normal_vector, phi)
    return pi - phi_dot_normal - (gamma2 * spacetime_metric)


# Test functions for evolved fields


def evol_field_psi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    return upsi


def evol_field_pi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    return 0.5 * (uplus + uminus) + gamma2 * upsi


def evol_field_phi(gamma2, upsi, uzero, uplus, uminus, normal_one_form):
    udiff = 0.5 * (uplus - uminus)
    return np.einsum("i,ab->iab", normal_one_form, udiff) + uzero


# End test functions for characteristic fields
