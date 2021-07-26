# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from Evolution.Systems.CurvedScalarWave.Characteristics import (
    char_speed_vpsi, char_speed_vzero, char_speed_vplus, char_speed_vminus,
    char_field_vpsi, char_field_vzero, char_field_vplus, char_field_vminus,
    evol_field_psi, evol_field_pi, evol_field_phi)


def dg_package_data_v_psi(pi, phi, psi, lapse, shift, inverse_spatial_metric,
                          constraint_gamma1, constraint_gamma2,
                          interface_unit_normal, interface_unit_normal_vector,
                          mesh_velocity, normal_dot_mesh_velocity):
    return char_field_vpsi(constraint_gamma2, inverse_spatial_metric, psi, pi,
                           phi, interface_unit_normal)


def dg_package_data_v_zero(pi, phi, psi, lapse, shift, inverse_spatial_metric,
                           constraint_gamma1, constraint_gamma2,
                           interface_unit_normal, interface_unit_normal_vector,
                           mesh_velocity, normal_dot_mesh_velocity):
    return char_field_vzero(constraint_gamma2, inverse_spatial_metric, psi, pi,
                            phi, interface_unit_normal)


def dg_package_data_v_plus(pi, phi, psi, lapse, shift, inverse_spatial_metric,
                           constraint_gamma1, constraint_gamma2,
                           interface_unit_normal, interface_unit_normal_vector,
                           mesh_velocity, normal_dot_mesh_velocity):
    return char_field_vplus(constraint_gamma2, inverse_spatial_metric, psi, pi,
                            phi, interface_unit_normal)


def dg_package_data_v_minus(pi, phi, psi, lapse, shift, inverse_spatial_metric,
                            constraint_gamma1, constraint_gamma2,
                            interface_unit_normal,
                            interface_unit_normal_vector, mesh_velocity,
                            normal_dot_mesh_velocity):
    return char_field_vminus(constraint_gamma2, inverse_spatial_metric, psi,
                             pi, phi, interface_unit_normal)


def dg_package_data_gamma2(pi, phi, psi, lapse, shift, inverse_spatial_metric,
                           constraint_gamma1, constraint_gamma2,
                           interface_unit_normal, interface_unit_normal_vector,
                           mesh_velocity, normal_dot_mesh_velocity):
    return constraint_gamma2


def dg_package_data_interface_unit_normal(
    pi, phi, psi, lapse, shift, inverse_spatial_metric, constraint_gamma1,
    constraint_gamma2, interface_unit_normal, interface_unit_normal_vector,
    mesh_velocity, normal_dot_mesh_velocity):
    return interface_unit_normal


def dg_package_data_char_speeds(pi, phi, psi, lapse, shift,
                                inverse_spatial_metric, constraint_gamma1,
                                constraint_gamma2, interface_unit_normal,
                                interface_unit_normal_vector, mesh_velocity,
                                normal_dot_mesh_velocity):
    result = np.zeros([4])
    result[0] = char_speed_vpsi(constraint_gamma1, lapse, shift,
                                interface_unit_normal)
    result[1] = char_speed_vzero(constraint_gamma1, lapse, shift,
                                 interface_unit_normal)
    result[2] = char_speed_vplus(constraint_gamma1, lapse, shift,
                                 interface_unit_normal)
    result[3] = char_speed_vminus(constraint_gamma1, lapse, shift,
                                  interface_unit_normal)
    if normal_dot_mesh_velocity is not None:
        result = result - normal_dot_mesh_velocity
    return result


def _weight_char_fields(v_psi_int, v_zero_int, v_plus_int, v_minus_int,
                        char_speeds_int, v_psi_ext, v_zero_ext, v_plus_ext,
                        v_minus_ext, char_speeds_ext):
    def weight(char_field, char_speed, sign):
        weighted_field = char_field
        weighted_field *= -sign * char_speed * np.heaviside(
            sign * char_speed, 0)
        return weighted_field

    return (weight(v_psi_int, char_speeds_int[0],
                   -1), weight(v_zero_int, char_speeds_int[1],
                               -1), weight(v_plus_int, char_speeds_int[2], -1),
            weight(v_minus_int, char_speeds_int[3],
                   -1), weight(v_psi_ext, char_speeds_ext[0],
                               1), weight(v_zero_ext, char_speeds_ext[1], 1),
            weight(v_plus_ext, char_speeds_ext[2],
                   1), weight(v_minus_ext, char_speeds_ext[3], 1))


def dg_boundary_terms_pi(v_psi_int, v_zero_int, v_plus_int, v_minus_int,
                         gamma2_int, interface_normal_int, char_speeds_int,
                         v_psi_ext, v_zero_ext, v_plus_ext, v_minus_ext,
                         gamma2_ext, interface_normal_ext, char_speeds_ext,
                         use_strong_form):
    (weighted_v_psi_int, weighted_v_zero_int, weighted_v_plus_int,
     weighted_v_minus_int, weighted_v_psi_ext, weighted_v_zero_ext,
     weighted_v_plus_ext, weighted_v_minus_ext) = _weight_char_fields(
         v_psi_int, v_zero_int, v_plus_int, v_minus_int, char_speeds_int,
         v_psi_ext, v_zero_ext, v_plus_ext, v_minus_ext, char_speeds_ext)
    return (evol_field_pi(gamma2_ext, weighted_v_psi_ext, weighted_v_zero_ext,
                          weighted_v_plus_ext, weighted_v_minus_ext,
                          interface_normal_ext) -
            evol_field_pi(gamma2_int, weighted_v_psi_int, weighted_v_zero_int,
                          weighted_v_plus_int, weighted_v_minus_int,
                          interface_normal_int))


def dg_boundary_terms_phi(v_psi_int, v_zero_int, v_plus_int, v_minus_int,
                          gamma2_int, interface_normal_int, char_speeds_int,
                          v_psi_ext, v_zero_ext, v_plus_ext, v_minus_ext,
                          gamma2_ext, interface_normal_ext, char_speeds_ext,
                          use_strong_form):
    (weighted_v_psi_int, weighted_v_zero_int, weighted_v_plus_int,
     weighted_v_minus_int, weighted_v_psi_ext, weighted_v_zero_ext,
     weighted_v_plus_ext, weighted_v_minus_ext) = _weight_char_fields(
         v_psi_int, v_zero_int, v_plus_int, v_minus_int, char_speeds_int,
         v_psi_ext, v_zero_ext, v_plus_ext, v_minus_ext, char_speeds_ext)
    return (evol_field_phi(gamma2_ext, weighted_v_psi_ext, weighted_v_zero_ext,
                           weighted_v_plus_ext, weighted_v_minus_ext,
                           interface_normal_ext) -
            evol_field_phi(gamma2_int, weighted_v_psi_int, weighted_v_zero_int,
                           weighted_v_plus_int, weighted_v_minus_int,
                           interface_normal_int))


def dg_boundary_terms_psi(v_psi_int, v_zero_int, v_plus_int, v_minus_int,
                          gamma2_int, interface_normal_int, char_speeds_int,
                          v_psi_ext, v_zero_ext, v_plus_ext, v_minus_ext,
                          gamma2_ext, interface_normal_ext, char_speeds_ext,
                          use_strong_form):
    (weighted_v_psi_int, weighted_v_zero_int, weighted_v_plus_int,
     weighted_v_minus_int, weighted_v_psi_ext, weighted_v_zero_ext,
     weighted_v_plus_ext, weighted_v_minus_ext) = _weight_char_fields(
         v_psi_int, v_zero_int, v_plus_int, v_minus_int, char_speeds_int,
         v_psi_ext, v_zero_ext, v_plus_ext, v_minus_ext, char_speeds_ext)
    return (evol_field_psi(gamma2_ext, weighted_v_psi_ext, weighted_v_zero_ext,
                           weighted_v_plus_ext, weighted_v_minus_ext,
                           interface_normal_ext) -
            evol_field_psi(gamma2_int, weighted_v_psi_int, weighted_v_zero_int,
                           weighted_v_plus_int, weighted_v_minus_int,
                           interface_normal_int))
