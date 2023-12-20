# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    spacetime_metric,
    pi,
    phi,
    constraint_gamma1,
    constraint_gamma2,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    char_speeds = np.zeros([4])
    char_speeds[0] = -(1.0 + constraint_gamma1) * np.dot(shift, normal_covector)
    char_speeds[1] = -np.dot(shift, normal_covector)
    char_speeds[2] = lapse - np.dot(shift, normal_covector)
    char_speeds[3] = -lapse - np.dot(shift, normal_covector)
    if not (normal_dot_mesh_velocity is None):
        for i in range(4):
            char_speeds[i] -= normal_dot_mesh_velocity
        char_speeds[0] -= normal_dot_mesh_velocity * constraint_gamma1

    char_speed_v_zero = (
        phi - np.einsum("j,i,jab->iab", normal_vector, normal_covector, phi)
    ) * char_speeds[1]
    char_speed_v_plus = (
        pi
        + np.einsum("iab,i->ab", phi, normal_vector)
        - constraint_gamma2 * spacetime_metric
    ) * char_speeds[2]

    char_speed_v_minus = (
        pi
        - np.einsum("iab,i->ab", phi, normal_vector)
        - constraint_gamma2 * spacetime_metric
    ) * char_speeds[3]

    return (
        np.asarray(char_speeds[0] * spacetime_metric),
        char_speed_v_zero,
        char_speed_v_plus,
        char_speed_v_minus,
        np.einsum("i,ab->iab", normal_covector, char_speed_v_plus),
        np.einsum("i,ab->iab", normal_covector, char_speed_v_minus),
        np.asarray(char_speeds[0] * spacetime_metric * constraint_gamma2),
        char_speeds,
    )


def dg_boundary_terms(
    int_char_speed_v_spacetime_metric,
    int_char_speed_v_zero,
    int_char_speed_v_plus,
    int_char_speed_v_minus,
    int_char_speed_v_plus_times_normal,
    int_char_speed_v_minus_times_normal,
    int_char_speed_gamma2_v_spacetime_metric,
    int_char_speeds,
    ext_char_speed_v_spacetime_metric,
    ext_char_speed_v_zero,
    ext_char_speed_v_plus,
    ext_char_speed_v_minus,
    ext_char_speed_v_plus_times_normal,
    ext_char_speed_v_minus_times_normal,
    ext_char_speed_gamma2_v_spacetime_metric,
    ext_char_speeds,
    use_strong_form,
):
    result_spacetime_metric = int_char_speed_v_spacetime_metric * 0.0
    if ext_char_speeds[0] > 0.0:
        result_spacetime_metric -= ext_char_speed_v_spacetime_metric
    if int_char_speeds[0] < 0.0:
        result_spacetime_metric -= int_char_speed_v_spacetime_metric

    result_pi = int_char_speed_v_spacetime_metric * 0.0
    # Add v^+ terms
    if ext_char_speeds[2] > 0.0:
        result_pi -= 0.5 * ext_char_speed_v_plus
    if int_char_speeds[2] < 0.0:
        result_pi -= 0.5 * int_char_speed_v_plus

    # Add v^- terms
    if ext_char_speeds[3] > 0.0:
        result_pi -= 0.5 * ext_char_speed_v_minus
    if int_char_speeds[3] < 0.0:
        result_pi -= 0.5 * int_char_speed_v_minus

    # Add v^\Psi terms
    if ext_char_speeds[0] > 0.0:
        result_pi -= ext_char_speed_gamma2_v_spacetime_metric
    if int_char_speeds[0] < 0.0:
        result_pi -= int_char_speed_gamma2_v_spacetime_metric

    result_phi = int_char_speed_v_zero * 0.0

    # Add v^+ terms
    if ext_char_speeds[2] > 0.0:
        result_phi -= 0.5 * ext_char_speed_v_plus_times_normal
    if int_char_speeds[2] < 0.0:
        result_phi -= 0.5 * int_char_speed_v_plus_times_normal

    # Add v^- terms
    if ext_char_speeds[3] > 0.0:
        result_phi += 0.5 * ext_char_speed_v_minus_times_normal
    if int_char_speeds[3] < 0.0:
        result_phi += 0.5 * int_char_speed_v_minus_times_normal

    # Add v^0 terms
    if ext_char_speeds[1] > 0.0:
        result_phi -= ext_char_speed_v_zero
    if int_char_speeds[1] < 0.0:
        result_phi -= int_char_speed_v_zero

    return (
        result_spacetime_metric,
        result_pi,
        result_phi,
    )
