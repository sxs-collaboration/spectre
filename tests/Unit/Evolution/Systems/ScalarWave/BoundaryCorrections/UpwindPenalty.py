# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data_char_speed_v_psi(pi, phi, psi, constraint_gamma2,
                                     normal_covector, mesh_velocity,
                                     normal_dot_mesh_velocity):
    if normal_dot_mesh_velocity is None:
        return 0.0 * psi
    else:
        return -normal_dot_mesh_velocity * psi


def dg_package_data_char_speed_v_zero(pi, phi, psi, constraint_gamma2,
                                      normal_covector, mesh_velocity,
                                      normal_dot_mesh_velocity):
    if normal_dot_mesh_velocity is None:
        return 0.0 * phi
    else:
        normal_dot_phi = np.dot(phi, normal_covector)
        result = phi
        for i in range(len(phi)):
            result[i] -= normal_covector[i] * normal_dot_phi
        char_speed = -normal_dot_mesh_velocity
        result *= char_speed
        return result


def dg_package_data_char_speed_v_plus(pi, phi, psi, constraint_gamma2,
                                      normal_covector, mesh_velocity,
                                      normal_dot_mesh_velocity):
    result = pi + np.dot(phi, normal_covector) - constraint_gamma2 * psi
    result *= (1.0 if normal_dot_mesh_velocity is None else 1.0 -
               normal_dot_mesh_velocity)
    return result


def dg_package_data_char_speed_v_minus(pi, phi, psi, constraint_gamma2,
                                       normal_covector, mesh_velocity,
                                       normal_dot_mesh_velocity):
    result = pi - np.dot(phi, normal_covector) - constraint_gamma2 * psi
    result *= (-1.0 if normal_dot_mesh_velocity is None else -1.0 -
               normal_dot_mesh_velocity)
    return result


def dg_package_data_char_speed_v_plus_times_normal(pi, phi, psi,
                                                   constraint_gamma2,
                                                   normal_covector,
                                                   mesh_velocity,
                                                   normal_dot_mesh_velocity):
    return normal_covector * dg_package_data_char_speed_v_plus(
        pi, phi, psi, constraint_gamma2, normal_covector, mesh_velocity,
        normal_dot_mesh_velocity)


def dg_package_data_char_speed_v_minus_times_normal(pi, phi, psi,
                                                    constraint_gamma2,
                                                    normal_covector,
                                                    mesh_velocity,
                                                    normal_dot_mesh_velocity):
    return normal_covector * dg_package_data_char_speed_v_minus(
        pi, phi, psi, constraint_gamma2, normal_covector, mesh_velocity,
        normal_dot_mesh_velocity)


def dg_package_data_char_speed_gamma2_v_psi(pi, phi, psi, constraint_gamma2,
                                            normal_covector, mesh_velocity,
                                            normal_dot_mesh_velocity):
    if normal_dot_mesh_velocity is None:
        return 0.0 * psi * constraint_gamma2
    else:
        return -normal_dot_mesh_velocity * psi * constraint_gamma2


def dg_package_data_char_speeds(pi, phi, psi, constraint_gamma2,
                                normal_covector, mesh_velocity,
                                normal_dot_mesh_velocity):
    result = np.zeros([3])
    result[0] = (0.0 if normal_dot_mesh_velocity is None else
                 -normal_dot_mesh_velocity)
    result[1] = (1.0 if normal_dot_mesh_velocity is None else 1.0 -
                 normal_dot_mesh_velocity)
    result[2] = (-1.0 if normal_dot_mesh_velocity is None else -1.0 -
                 normal_dot_mesh_velocity)
    return result


def dg_boundary_terms_pi(
    int_char_speed_v_psi, int_char_speed_v_zero, int_char_speed_v_plus,
    int_char_speed_v_minus, int_char_speed_v_plus_times_normal,
    int_char_speed_v_minus_times_normal, int_char_speed_gamma2_v_psi,
    int_char_speeds, ext_char_speed_v_psi, ext_char_speed_v_zero,
    ext_char_speed_v_plus, ext_char_speed_v_minus,
    ext_char_speed_v_plus_times_normal, ext_char_speed_v_minus_times_normal,
    ext_char_speed_gamma2_v_psi, ext_char_speeds, use_strong_form):
    result = int_char_speed_v_psi * 0.
    # Add v^+ terms
    if ext_char_speeds[1] > 0.:
        result -= 0.5 * ext_char_speed_v_plus
    if int_char_speeds[1] < 0.:
        result -= 0.5 * int_char_speed_v_plus

    # Add v^- terms
    if ext_char_speeds[2] > 0.:
        result -= 0.5 * ext_char_speed_v_minus
    if int_char_speeds[2] < 0.:
        result -= 0.5 * int_char_speed_v_minus

    # Add v^\Psi terms
    if ext_char_speeds[0] > 0.:
        result -= ext_char_speed_gamma2_v_psi
    if int_char_speeds[0] < 0.:
        result -= int_char_speed_gamma2_v_psi
    return result


def dg_boundary_terms_phi(
    int_char_speed_v_psi, int_char_speed_v_zero, int_char_speed_v_plus,
    int_char_speed_v_minus, int_char_speed_v_plus_times_normal,
    int_char_speed_v_minus_times_normal, int_char_speed_gamma2_v_psi,
    int_char_speeds, ext_char_speed_v_psi, ext_char_speed_v_zero,
    ext_char_speed_v_plus, ext_char_speed_v_minus,
    ext_char_speed_v_plus_times_normal, ext_char_speed_v_minus_times_normal,
    ext_char_speed_gamma2_v_psi, ext_char_speeds, use_strong_form):
    result = int_char_speed_v_zero * 0.

    # Add v^+ terms
    if ext_char_speeds[1] >= 0.:
        result -= 0.5 * ext_char_speed_v_plus_times_normal
    if int_char_speeds[1] < 0.:
        result -= 0.5 * int_char_speed_v_plus_times_normal

    # Add v^- terms
    if ext_char_speeds[2] >= 0.:
        result += 0.5 * ext_char_speed_v_minus_times_normal
    if int_char_speeds[2] < 0.:
        result += 0.5 * int_char_speed_v_minus_times_normal

    # Add v^0 terms
    if ext_char_speeds[0] >= 0.:
        result -= ext_char_speed_v_zero
    if int_char_speeds[0] < 0.:
        result -= int_char_speed_v_zero
    return result


def dg_boundary_terms_psi(
    int_char_speed_v_psi, int_char_speed_v_zero, int_char_speed_v_plus,
    int_char_speed_v_minus, int_char_speed_v_plus_times_normal,
    int_char_speed_v_minus_times_normal, int_char_speed_gamma2_v_psi,
    int_char_speeds, ext_char_speed_v_psi, ext_char_speed_v_zero,
    ext_char_speed_v_plus, ext_char_speed_v_minus,
    ext_char_speed_v_plus_times_normal, ext_char_speed_v_minus_times_normal,
    ext_char_speed_gamma2_v_psi, ext_char_speeds, use_strong_form):
    result = int_char_speed_v_psi * 0.
    if ext_char_speeds[0] >= 0.:
        result -= ext_char_speed_v_psi
    if int_char_speeds[0] < 0.:
        result -= int_char_speed_v_psi
    return result
