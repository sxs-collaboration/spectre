# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    psi,
    pi,
    phi,
    constraint_gamma2,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    char_speeds = np.zeros([3])
    char_speeds[0] = (
        0.0 if normal_dot_mesh_velocity is None else -normal_dot_mesh_velocity
    )
    char_speeds[1] = (
        1.0
        if normal_dot_mesh_velocity is None
        else 1.0 - normal_dot_mesh_velocity
    )
    char_speeds[2] = (
        -1.0
        if normal_dot_mesh_velocity is None
        else -1.0 - normal_dot_mesh_velocity
    )

    char_speed_v_plus = np.asarray(
        (pi + np.dot(phi, normal_covector) - constraint_gamma2 * psi)
        * char_speeds[1]
    )
    char_speed_v_minus = np.asarray(
        (pi - np.dot(phi, normal_covector) - constraint_gamma2 * psi)
        * char_speeds[2]
    )

    return (
        np.asarray(char_speeds[0] * psi),
        np.asarray(
            0.0 * phi
            if normal_dot_mesh_velocity is None
            else -normal_dot_mesh_velocity
            * (phi - np.einsum("ijj->i", normal_covector, normal_covector, phi))
        ),
        char_speed_v_plus,
        char_speed_v_minus,
        normal_covector * char_speed_v_plus,
        normal_covector * char_speed_v_minus,
        np.asarray(char_speeds[0] * psi * constraint_gamma2),
        char_speeds,
    )


def dg_boundary_terms(
    int_char_speed_v_psi,
    int_char_speed_v_zero,
    int_char_speed_v_plus,
    int_char_speed_v_minus,
    int_char_speed_v_plus_times_normal,
    int_char_speed_v_minus_times_normal,
    int_char_speed_gamma2_v_psi,
    int_char_speeds,
    ext_char_speed_v_psi,
    ext_char_speed_v_zero,
    ext_char_speed_v_plus,
    ext_char_speed_v_minus,
    ext_char_speed_v_plus_times_normal,
    ext_char_speed_v_minus_times_normal,
    ext_char_speed_gamma2_v_psi,
    ext_char_speeds,
    use_strong_form,
):
    result_psi = int_char_speed_v_psi * 0.0
    if ext_char_speeds[0] >= 0.0:
        result_psi -= ext_char_speed_v_psi
    if int_char_speeds[0] < 0.0:
        result_psi -= int_char_speed_v_psi

    result_pi = int_char_speed_v_psi * 0.0
    # Add v^+ terms
    if ext_char_speeds[1] > 0.0:
        result_pi -= 0.5 * ext_char_speed_v_plus
    if int_char_speeds[1] < 0.0:
        result_pi -= 0.5 * int_char_speed_v_plus

    # Add v^- terms
    if ext_char_speeds[2] > 0.0:
        result_pi -= 0.5 * ext_char_speed_v_minus
    if int_char_speeds[2] < 0.0:
        result_pi -= 0.5 * int_char_speed_v_minus

    # Add v^\Psi terms
    if ext_char_speeds[0] > 0.0:
        result_pi -= ext_char_speed_gamma2_v_psi
    if int_char_speeds[0] < 0.0:
        result_pi -= int_char_speed_gamma2_v_psi

    result_phi = int_char_speed_v_zero * 0.0

    # Add v^+ terms
    if ext_char_speeds[1] >= 0.0:
        result_phi -= 0.5 * ext_char_speed_v_plus_times_normal
    if int_char_speeds[1] < 0.0:
        result_phi -= 0.5 * int_char_speed_v_plus_times_normal

    # Add v^- terms
    if ext_char_speeds[2] >= 0.0:
        result_phi += 0.5 * ext_char_speed_v_minus_times_normal
    if int_char_speeds[2] < 0.0:
        result_phi += 0.5 * int_char_speed_v_minus_times_normal

    # Add v^0 terms
    if ext_char_speeds[0] >= 0.0:
        result_phi -= ext_char_speed_v_zero
    if int_char_speeds[0] < 0.0:
        result_phi -= int_char_speed_v_zero

    return (
        np.asarray(result_psi),
        np.asarray(result_pi),
        np.asarray(result_phi),
    )
