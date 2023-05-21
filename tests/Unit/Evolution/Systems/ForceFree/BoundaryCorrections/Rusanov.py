# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data_tilde_e(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return tilde_e


def dg_package_data_tilde_b(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return tilde_b


def dg_package_data_tilde_psi(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return tilde_psi


def dg_package_data_tilde_phi(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return tilde_phi


def dg_package_data_tilde_q(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return tilde_q


def dg_package_data_normal_dot_flux_tilde_e(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return np.einsum("ij,i->j", flux_tilde_e, normal_covector)


def dg_package_data_normal_dot_flux_tilde_b(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return np.einsum("ij,i->j", flux_tilde_b, normal_covector)


def dg_package_data_normal_dot_flux_tilde_psi(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return np.dot(flux_tilde_psi, normal_covector)


def dg_package_data_normal_dot_flux_tilde_phi(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return np.dot(flux_tilde_phi, normal_covector)


def dg_package_data_normal_dot_flux_tilde_q(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return np.dot(flux_tilde_q, normal_covector)


def dg_package_data_abs_char_speed(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    if normal_dot_mesh_velocity is None:
        return np.maximum(
            np.abs(lapse - np.dot(shift, normal_covector)),
            np.abs(-lapse - np.dot(shift, normal_covector)),
        )
    else:
        return np.maximum(
            np.abs(
                lapse
                - np.dot(shift, normal_covector)
                - normal_dot_mesh_velocity
            ),
            np.abs(
                -lapse
                - np.dot(shift, normal_covector)
                - normal_dot_mesh_velocity
            ),
        )


def dg_boundary_terms_tilde_e(
    interior_tilde_e,
    interior_tilde_b,
    interior_tilde_psi,
    interior_tilde_phi,
    interior_tilde_q,
    interior_normal_dot_flux_tilde_e,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_psi,
    interior_normal_dot_flux_tilde_phi,
    interior_normal_dot_flux_tilde_q,
    interior_abs_char_speed,
    exterior_tilde_e,
    exterior_tilde_b,
    exterior_tilde_psi,
    exterior_tilde_phi,
    exterior_tilde_q,
    exterior_normal_dot_flux_tilde_e,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_psi,
    exterior_normal_dot_flux_tilde_phi,
    exterior_normal_dot_flux_tilde_q,
    exterior_abs_char_speed,
    use_strong_form,
):
    if use_strong_form:
        return -0.5 * (
            interior_normal_dot_flux_tilde_e + exterior_normal_dot_flux_tilde_e
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_e - interior_tilde_e
        )
    else:
        return 0.5 * (
            interior_normal_dot_flux_tilde_e - exterior_normal_dot_flux_tilde_e
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_e - interior_tilde_e
        )


def dg_boundary_terms_tilde_b(
    interior_tilde_e,
    interior_tilde_b,
    interior_tilde_psi,
    interior_tilde_phi,
    interior_tilde_q,
    interior_normal_dot_flux_tilde_e,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_psi,
    interior_normal_dot_flux_tilde_phi,
    interior_normal_dot_flux_tilde_q,
    interior_abs_char_speed,
    exterior_tilde_e,
    exterior_tilde_b,
    exterior_tilde_psi,
    exterior_tilde_phi,
    exterior_tilde_q,
    exterior_normal_dot_flux_tilde_e,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_psi,
    exterior_normal_dot_flux_tilde_phi,
    exterior_normal_dot_flux_tilde_q,
    exterior_abs_char_speed,
    use_strong_form,
):
    if use_strong_form:
        return -0.5 * (
            interior_normal_dot_flux_tilde_b + exterior_normal_dot_flux_tilde_b
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_b - interior_tilde_b
        )
    else:
        return 0.5 * (
            interior_normal_dot_flux_tilde_b - exterior_normal_dot_flux_tilde_b
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_b - interior_tilde_b
        )


def dg_boundary_terms_tilde_psi(
    interior_tilde_e,
    interior_tilde_b,
    interior_tilde_psi,
    interior_tilde_phi,
    interior_tilde_q,
    interior_normal_dot_flux_tilde_e,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_psi,
    interior_normal_dot_flux_tilde_phi,
    interior_normal_dot_flux_tilde_q,
    interior_abs_char_speed,
    exterior_tilde_e,
    exterior_tilde_b,
    exterior_tilde_psi,
    exterior_tilde_phi,
    exterior_tilde_q,
    exterior_normal_dot_flux_tilde_e,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_psi,
    exterior_normal_dot_flux_tilde_phi,
    exterior_normal_dot_flux_tilde_q,
    exterior_abs_char_speed,
    use_strong_form,
):
    if use_strong_form:
        return -0.5 * (
            interior_normal_dot_flux_tilde_psi
            + exterior_normal_dot_flux_tilde_psi
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_psi - interior_tilde_psi
        )
    else:
        return 0.5 * (
            interior_normal_dot_flux_tilde_psi
            - exterior_normal_dot_flux_tilde_psi
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_psi - interior_tilde_psi
        )


def dg_boundary_terms_tilde_phi(
    interior_tilde_e,
    interior_tilde_b,
    interior_tilde_psi,
    interior_tilde_phi,
    interior_tilde_q,
    interior_normal_dot_flux_tilde_e,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_psi,
    interior_normal_dot_flux_tilde_phi,
    interior_normal_dot_flux_tilde_q,
    interior_abs_char_speed,
    exterior_tilde_e,
    exterior_tilde_b,
    exterior_tilde_psi,
    exterior_tilde_phi,
    exterior_tilde_q,
    exterior_normal_dot_flux_tilde_e,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_psi,
    exterior_normal_dot_flux_tilde_phi,
    exterior_normal_dot_flux_tilde_q,
    exterior_abs_char_speed,
    use_strong_form,
):
    if use_strong_form:
        return -0.5 * (
            interior_normal_dot_flux_tilde_phi
            + exterior_normal_dot_flux_tilde_phi
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_phi - interior_tilde_phi
        )
    else:
        return 0.5 * (
            interior_normal_dot_flux_tilde_phi
            - exterior_normal_dot_flux_tilde_phi
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_phi - interior_tilde_phi
        )


def dg_boundary_terms_tilde_q(
    interior_tilde_e,
    interior_tilde_b,
    interior_tilde_psi,
    interior_tilde_phi,
    interior_tilde_q,
    interior_normal_dot_flux_tilde_e,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_psi,
    interior_normal_dot_flux_tilde_phi,
    interior_normal_dot_flux_tilde_q,
    interior_abs_char_speed,
    exterior_tilde_e,
    exterior_tilde_b,
    exterior_tilde_psi,
    exterior_tilde_phi,
    exterior_tilde_q,
    exterior_normal_dot_flux_tilde_e,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_psi,
    exterior_normal_dot_flux_tilde_phi,
    exterior_normal_dot_flux_tilde_q,
    exterior_abs_char_speed,
    use_strong_form,
):
    if use_strong_form:
        return -0.5 * (
            interior_normal_dot_flux_tilde_q + exterior_normal_dot_flux_tilde_q
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_q - interior_tilde_q
        )
    else:
        return 0.5 * (
            interior_normal_dot_flux_tilde_q - exterior_normal_dot_flux_tilde_q
        ) - 0.5 * np.maximum(
            interior_abs_char_speed, exterior_abs_char_speed
        ) * (
            exterior_tilde_q - interior_tilde_q
        )
