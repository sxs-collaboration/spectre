# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Evolution.Systems.CurvedScalarWave.Characteristics import (
    char_field_vminus,
    char_field_vplus,
    char_field_vpsi,
    char_field_vzero,
    char_speed_vminus,
    char_speed_vplus,
    char_speed_vpsi,
    char_speed_vzero,
    evol_field_phi,
    evol_field_pi,
    evol_field_psi,
)


def dg_package_data(
    psi,
    pi,
    phi,
    lapse,
    shift,
    constraint_gamma1,
    constraint_gamma2,
    interface_unit_normal,
    interface_unit_normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    char_speeds = np.zeros([4])
    char_speeds[0] = char_speed_vpsi(
        constraint_gamma1, lapse, shift, interface_unit_normal
    )
    char_speeds[1] = char_speed_vzero(
        constraint_gamma1, lapse, shift, interface_unit_normal
    )
    char_speeds[2] = char_speed_vplus(
        constraint_gamma1, lapse, shift, interface_unit_normal
    )
    char_speeds[3] = char_speed_vminus(
        constraint_gamma1, lapse, shift, interface_unit_normal
    )
    if normal_dot_mesh_velocity is not None:
        char_speeds = char_speeds - normal_dot_mesh_velocity

    return (
        char_field_vpsi(
            constraint_gamma2,
            psi,
            pi,
            phi,
            interface_unit_normal,
            interface_unit_normal_vector,
        ),
        char_field_vzero(
            constraint_gamma2,
            psi,
            pi,
            phi,
            interface_unit_normal,
            interface_unit_normal_vector,
        ),
        np.asarray(
            char_field_vplus(
                constraint_gamma2,
                psi,
                pi,
                phi,
                interface_unit_normal,
                interface_unit_normal_vector,
            )
        ),
        np.asarray(
            char_field_vminus(
                constraint_gamma2,
                psi,
                pi,
                phi,
                interface_unit_normal,
                interface_unit_normal_vector,
            )
        ),
        np.asarray(constraint_gamma2),
        np.asarray(interface_unit_normal),
        char_speeds,
    )


def _weight_char_fields(
    v_psi_int,
    v_zero_int,
    v_plus_int,
    v_minus_int,
    char_speeds_int,
    v_psi_ext,
    v_zero_ext,
    v_plus_ext,
    v_minus_ext,
    char_speeds_ext,
):
    def weight(char_field, char_speed, sign):
        weighted_field = char_field
        weighted_field *= (
            -sign * char_speed * np.heaviside(sign * char_speed, 0)
        )
        return weighted_field

    return (
        weight(v_psi_int, char_speeds_int[0], -1),
        weight(v_zero_int, char_speeds_int[1], -1),
        weight(v_plus_int, char_speeds_int[2], -1),
        weight(v_minus_int, char_speeds_int[3], -1),
        weight(v_psi_ext, char_speeds_ext[0], 1),
        weight(v_zero_ext, char_speeds_ext[1], 1),
        weight(v_plus_ext, char_speeds_ext[2], 1),
        weight(v_minus_ext, char_speeds_ext[3], 1),
    )


def dg_boundary_terms(
    v_psi_int,
    v_zero_int,
    v_plus_int,
    v_minus_int,
    gamma2_int,
    interface_normal_int,
    char_speeds_int,
    v_psi_ext,
    v_zero_ext,
    v_plus_ext,
    v_minus_ext,
    gamma2_ext,
    interface_normal_ext,
    char_speeds_ext,
    use_strong_form,
):
    (
        weighted_v_psi_int,
        weighted_v_zero_int,
        weighted_v_plus_int,
        weighted_v_minus_int,
        weighted_v_psi_ext,
        weighted_v_zero_ext,
        weighted_v_plus_ext,
        weighted_v_minus_ext,
    ) = _weight_char_fields(
        v_psi_int,
        v_zero_int,
        v_plus_int,
        v_minus_int,
        char_speeds_int,
        v_psi_ext,
        v_zero_ext,
        v_plus_ext,
        v_minus_ext,
        char_speeds_ext,
    )
    return (
        np.asarray(
            evol_field_psi(
                gamma2_int,
                weighted_v_psi_ext,
                weighted_v_zero_ext,
                weighted_v_plus_ext,
                weighted_v_minus_ext,
                -interface_normal_int,
            )
            - evol_field_psi(
                gamma2_int,
                weighted_v_psi_int,
                weighted_v_zero_int,
                weighted_v_plus_int,
                weighted_v_minus_int,
                interface_normal_int,
            )
        ),
        np.asarray(
            evol_field_pi(
                gamma2_int,
                weighted_v_psi_ext,
                weighted_v_zero_ext,
                weighted_v_plus_ext,
                weighted_v_minus_ext,
                -interface_normal_int,
            )
            - evol_field_pi(
                gamma2_int,
                weighted_v_psi_int,
                weighted_v_zero_int,
                weighted_v_plus_int,
                weighted_v_minus_int,
                interface_normal_int,
            )
        ),
        np.asarray(
            evol_field_phi(
                gamma2_int,
                weighted_v_psi_ext,
                weighted_v_zero_ext,
                weighted_v_plus_ext,
                weighted_v_minus_ext,
                -interface_normal_int,
            )
            - evol_field_phi(
                gamma2_int,
                weighted_v_psi_int,
                weighted_v_zero_int,
                weighted_v_plus_int,
                weighted_v_minus_int,
                interface_normal_int,
            )
        ),
    )
