# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data_char_speed_v_spacetime_metric(spacetime_metric, pi, phi,
                                                  constraint_gamma1,
                                                  constraint_gamma2, lapse,
                                                  shift, normal_covector,
                                                  normal_vector, mesh_velocity,
                                                  normal_dot_mesh_velocity):
    if normal_dot_mesh_velocity is None:
        return -(1.0 + constraint_gamma1) * np.dot(
            shift, normal_covector) * spacetime_metric
    else:
        return (-(1.0 + constraint_gamma1) * np.dot(shift, normal_covector) -
                normal_dot_mesh_velocity) * spacetime_metric


def dg_package_data_char_speed_v_zero(spacetime_metric, pi, phi,
                                      constraint_gamma1, constraint_gamma2,
                                      lapse, shift, normal_covector,
                                      normal_vector, mesh_velocity,
                                      normal_dot_mesh_velocity):
    result = phi - np.einsum("j,i,jab->iab", normal_vector, normal_covector,
                             phi)

    if normal_dot_mesh_velocity is None:
        return -np.dot(shift, normal_covector) * result
    else:
        return (-np.dot(shift, normal_covector) -
                normal_dot_mesh_velocity) * result


def dg_package_data_char_speed_v_plus(spacetime_metric, pi, phi,
                                      constraint_gamma1, constraint_gamma2,
                                      lapse, shift, normal_covector,
                                      normal_vector, mesh_velocity,
                                      normal_dot_mesh_velocity):
    result = pi + np.einsum(
        "iab,i->ab", phi, normal_vector) - constraint_gamma2 * spacetime_metric
    if normal_dot_mesh_velocity is None:
        return (lapse - np.dot(shift, normal_covector)) * result
    else:
        return (lapse - np.dot(shift, normal_covector) -
                normal_dot_mesh_velocity) * result


def dg_package_data_char_speed_v_minus(spacetime_metric, pi, phi,
                                       constraint_gamma1, constraint_gamma2,
                                       lapse, shift, normal_covector,
                                       normal_vector, mesh_velocity,
                                       normal_dot_mesh_velocity):
    result = pi - np.einsum(
        "iab,i->ab", phi, normal_vector) - constraint_gamma2 * spacetime_metric
    if normal_dot_mesh_velocity is None:
        return (-lapse - np.dot(shift, normal_covector)) * result
    else:
        return (-lapse - np.dot(shift, normal_covector) -
                normal_dot_mesh_velocity) * result


def dg_package_data_char_speed_v_plus_times_normal(
    spacetime_metric, pi, phi, constraint_gamma1, constraint_gamma2, lapse,
    shift, normal_covector, normal_vector, mesh_velocity,
    normal_dot_mesh_velocity):
    return np.einsum(
        "i,ab->iab", normal_covector,
        dg_package_data_char_speed_v_plus(spacetime_metric, pi, phi,
                                          constraint_gamma1, constraint_gamma2,
                                          lapse, shift, normal_covector,
                                          normal_vector, mesh_velocity,
                                          normal_dot_mesh_velocity))


def dg_package_data_char_speed_v_minus_times_normal(
    spacetime_metric, pi, phi, constraint_gamma1, constraint_gamma2, lapse,
    shift, normal_covector, normal_vector, mesh_velocity,
    normal_dot_mesh_velocity):
    return np.einsum(
        "i,ab->iab", normal_covector,
        dg_package_data_char_speed_v_minus(spacetime_metric, pi, phi,
                                           constraint_gamma1,
                                           constraint_gamma2, lapse, shift,
                                           normal_covector, normal_vector,
                                           mesh_velocity,
                                           normal_dot_mesh_velocity))


def dg_package_data_char_speed_gamma2_v_spacetime_metric(
    spacetime_metric, pi, phi, constraint_gamma1, constraint_gamma2, lapse,
    shift, normal_covector, normal_vector, mesh_velocity,
    normal_dot_mesh_velocity):
    if normal_dot_mesh_velocity is None:
        return -(1.0 + constraint_gamma1) * np.dot(
            shift, normal_covector) * spacetime_metric * constraint_gamma2
    else:
        return (
            -(1.0 + constraint_gamma1) * np.dot(shift, normal_covector) -
            normal_dot_mesh_velocity) * spacetime_metric * constraint_gamma2


def dg_package_data_char_speeds(spacetime_metric, pi, phi, constraint_gamma1,
                                constraint_gamma2, lapse, shift,
                                normal_covector, normal_vector, mesh_velocity,
                                normal_dot_mesh_velocity):
    result = np.zeros([4])
    result[0] = -(1.0 + constraint_gamma1) * np.dot(shift, normal_covector)
    result[1] = -np.dot(shift, normal_covector)
    result[2] = lapse - np.dot(shift, normal_covector)
    result[3] = -lapse - np.dot(shift, normal_covector)
    if not (normal_dot_mesh_velocity is None):
        for i in range(4):
            result[i] -= normal_dot_mesh_velocity
    return result


def dg_boundary_terms_spacetime_metric(
    int_char_speed_v_spacetime_metric, int_char_speed_v_zero,
    int_char_speed_v_plus, int_char_speed_v_minus,
    int_char_speed_v_plus_times_normal, int_char_speed_v_minus_times_normal,
    int_char_speed_gamma2_v_spacetime_metric, int_char_speeds,
    ext_char_speed_v_spacetime_metric, ext_char_speed_v_zero,
    ext_char_speed_v_plus, ext_char_speed_v_minus,
    ext_char_speed_v_plus_times_normal, ext_char_speed_v_minus_times_normal,
    ext_char_speed_gamma2_v_spacetime_metric, ext_char_speeds,
    use_strong_form):
    result = int_char_speed_v_spacetime_metric * 0.
    if ext_char_speeds[0] > 0.:
        result -= ext_char_speed_v_spacetime_metric
    if int_char_speeds[0] < 0.:
        result -= int_char_speed_v_spacetime_metric
    return result


def dg_boundary_terms_pi(
    int_char_speed_v_spacetime_metric, int_char_speed_v_zero,
    int_char_speed_v_plus, int_char_speed_v_minus,
    int_char_speed_v_plus_times_normal, int_char_speed_v_minus_times_normal,
    int_char_speed_gamma2_v_spacetime_metric, int_char_speeds,
    ext_char_speed_v_spacetime_metric, ext_char_speed_v_zero,
    ext_char_speed_v_plus, ext_char_speed_v_minus,
    ext_char_speed_v_plus_times_normal, ext_char_speed_v_minus_times_normal,
    ext_char_speed_gamma2_v_spacetime_metric, ext_char_speeds,
    use_strong_form):
    result = int_char_speed_v_spacetime_metric * 0.
    # Add v^+ terms
    if ext_char_speeds[2] > 0.:
        result -= 0.5 * ext_char_speed_v_plus
    if int_char_speeds[2] < 0.:
        result -= 0.5 * int_char_speed_v_plus

    # Add v^- terms
    if ext_char_speeds[3] > 0.:
        result -= 0.5 * ext_char_speed_v_minus
    if int_char_speeds[3] < 0.:
        result -= 0.5 * int_char_speed_v_minus

    # Add v^\Psi terms
    if ext_char_speeds[0] > 0.:
        result -= ext_char_speed_gamma2_v_spacetime_metric
    if int_char_speeds[0] < 0.:
        result -= int_char_speed_gamma2_v_spacetime_metric
    return result


def dg_boundary_terms_phi(
    int_char_speed_v_spacetime_metric, int_char_speed_v_zero,
    int_char_speed_v_plus, int_char_speed_v_minus,
    int_char_speed_v_plus_times_normal, int_char_speed_v_minus_times_normal,
    int_char_speed_gamma2_v_spacetime_metric, int_char_speeds,
    ext_char_speed_v_spacetime_metric, ext_char_speed_v_zero,
    ext_char_speed_v_plus, ext_char_speed_v_minus,
    ext_char_speed_v_plus_times_normal, ext_char_speed_v_minus_times_normal,
    ext_char_speed_gamma2_v_spacetime_metric, ext_char_speeds,
    use_strong_form):
    result = int_char_speed_v_zero * 0.

    # Add v^+ terms
    if ext_char_speeds[2] > 0.:
        result -= 0.5 * ext_char_speed_v_plus_times_normal
    if int_char_speeds[2] < 0.:
        result -= 0.5 * int_char_speed_v_plus_times_normal

    # Add v^- terms
    if ext_char_speeds[3] > 0.:
        result += 0.5 * ext_char_speed_v_minus_times_normal
    if int_char_speeds[3] < 0.:
        result += 0.5 * int_char_speed_v_minus_times_normal

    # Add v^0 terms
    if ext_char_speeds[1] > 0.:
        result -= ext_char_speed_v_zero
    if int_char_speeds[1] < 0.:
        result -= int_char_speed_v_zero
    return result
