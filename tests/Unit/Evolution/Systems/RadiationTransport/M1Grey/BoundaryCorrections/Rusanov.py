# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

_max_abs_char_speed = 1.0


def dg_package_data_tilde_e_nue(tilde_e_nue, tilde_e_bar_nue, tilde_s_nue,
                                tilde_s_bar_nue, flux_tilde_e_nue,
                                flux_tilde_e_bar_nue, flux_tilde_s_nue,
                                flux_tilde_s_bar_nue, normal_covector,
                                normal_vector, mesh_velocity,
                                normal_dot_mesh_velocity):
    return tilde_e_nue


def dg_package_data_tilde_e_bar_nue(tilde_e_nue, tilde_e_bar_nue, tilde_s_nue,
                                    tilde_s_bar_nue, flux_tilde_e_nue,
                                    flux_tilde_e_bar_nue, flux_tilde_s_nue,
                                    flux_tilde_s_bar_nue, normal_covector,
                                    normal_vector, mesh_velocity,
                                    normal_dot_mesh_velocity):
    return tilde_e_bar_nue


def dg_package_data_tilde_s_nue(tilde_e_nue, tilde_e_bar_nue, tilde_s_nue,
                                tilde_s_bar_nue, flux_tilde_e_nue,
                                flux_tilde_e_bar_nue, flux_tilde_s_nue,
                                flux_tilde_s_bar_nue, normal_covector,
                                normal_vector, mesh_velocity,
                                normal_dot_mesh_velocity):
    return tilde_s_nue


def dg_package_data_tilde_s_bar_nue(tilde_e_nue, tilde_e_bar_nue, tilde_s_nue,
                                    tilde_s_bar_nue, flux_tilde_e_nue,
                                    flux_tilde_e_bar_nue, flux_tilde_s_nue,
                                    flux_tilde_s_bar_nue, normal_covector,
                                    normal_vector, mesh_velocity,
                                    normal_dot_mesh_velocity):
    return tilde_s_bar_nue


def dg_package_data_normal_dot_flux_tilde_e_nue(
    tilde_e_nue, tilde_e_bar_nue, tilde_s_nue, tilde_s_bar_nue,
    flux_tilde_e_nue, flux_tilde_e_bar_nue, flux_tilde_s_nue,
    flux_tilde_s_bar_nue, normal_covector, normal_vector, mesh_velocity,
    normal_dot_mesh_velocity):
    return np.einsum("i,i", normal_covector, flux_tilde_e_nue)


def dg_package_data_normal_dot_flux_tilde_e_bar_nue(
    tilde_e_nue, tilde_e_bar_nue, tilde_s_nue, tilde_s_bar_nue,
    flux_tilde_e_nue, flux_tilde_e_bar_nue, flux_tilde_s_nue,
    flux_tilde_s_bar_nue, normal_covector, normal_vector, mesh_velocity,
    normal_dot_mesh_velocity):
    return np.einsum("i,i", normal_covector, flux_tilde_e_bar_nue)


def dg_package_data_normal_dot_flux_tilde_s_nue(
    tilde_e_nue, tilde_e_bar_nue, tilde_s_nue, tilde_s_bar_nue,
    flux_tilde_e_nue, flux_tilde_e_bar_nue, flux_tilde_s_nue,
    flux_tilde_s_bar_nue, normal_covector, normal_vector, mesh_velocity,
    normal_dot_mesh_velocity):
    return np.einsum("i,ij->j", normal_covector, flux_tilde_s_nue)


def dg_package_data_normal_dot_flux_tilde_s_bar_nue(
    tilde_e_nue, tilde_e_bar_nue, tilde_s_nue, tilde_s_bar_nue,
    flux_tilde_e_nue, flux_tilde_e_bar_nue, flux_tilde_s_nue,
    flux_tilde_s_bar_nue, normal_covector, normal_vector, mesh_velocity,
    normal_dot_mesh_velocity):
    return np.einsum("i,ij->j", normal_covector, flux_tilde_s_bar_nue)


def dg_boundary_terms_tilde_e_nue(
    interior_tilde_e_nue, interior_tilde_e_bar_nue, interior_tilde_s_nue,
    interior_tilde_s_bar_nue, interior_normal_dot_flux_tilde_e_nue,
    interior_normal_dot_flux_tilde_e_bar_nue,
    interior_normal_dot_flux_tilde_s_nue,
    interior_normal_dot_flux_tilde_s_bar_nue, exterior_tilde_e_nue,
    exterior_tilde_e_bar_nue, exterior_tilde_s_nue, exterior_tilde_s_bar_nue,
    exterior_normal_dot_flux_tilde_e_nue,
    exterior_normal_dot_flux_tilde_e_bar_nue,
    exterior_normal_dot_flux_tilde_s_nue,
    exterior_normal_dot_flux_tilde_s_bar_nue, use_strong_form):
    if use_strong_form:
        return (-0.5 * (interior_normal_dot_flux_tilde_e_nue +
                        exterior_normal_dot_flux_tilde_e_nue) -
                0.5 * _max_abs_char_speed *
                (exterior_tilde_e_nue - interior_tilde_e_nue))
    else:
        return (0.5 * (interior_normal_dot_flux_tilde_e_nue -
                       exterior_normal_dot_flux_tilde_e_nue) -
                0.5 * _max_abs_char_speed *
                (exterior_tilde_e_nue - interior_tilde_e_nue))


def dg_boundary_terms_tilde_e_bar_nue(
    interior_tilde_e_nue, interior_tilde_e_bar_nue, interior_tilde_s_nue,
    interior_tilde_s_bar_nue, interior_normal_dot_flux_tilde_e_nue,
    interior_normal_dot_flux_tilde_e_bar_nue,
    interior_normal_dot_flux_tilde_s_nue,
    interior_normal_dot_flux_tilde_s_bar_nue, exterior_tilde_e_nue,
    exterior_tilde_e_bar_nue, exterior_tilde_s_nue, exterior_tilde_s_bar_nue,
    exterior_normal_dot_flux_tilde_e_nue,
    exterior_normal_dot_flux_tilde_e_bar_nue,
    exterior_normal_dot_flux_tilde_s_nue,
    exterior_normal_dot_flux_tilde_s_bar_nue, use_strong_form):
    if use_strong_form:
        return (-0.5 * (interior_normal_dot_flux_tilde_e_bar_nue +
                        exterior_normal_dot_flux_tilde_e_bar_nue) -
                0.5 * _max_abs_char_speed *
                (exterior_tilde_e_bar_nue - interior_tilde_e_bar_nue))
    else:
        return (0.5 * (interior_normal_dot_flux_tilde_e_bar_nue -
                       exterior_normal_dot_flux_tilde_e_bar_nue) -
                0.5 * _max_abs_char_speed *
                (exterior_tilde_e_bar_nue - interior_tilde_e_bar_nue))


def dg_boundary_terms_tilde_s_nue(
    interior_tilde_e_nue, interior_tilde_e_bar_nue, interior_tilde_s_nue,
    interior_tilde_s_bar_nue, interior_normal_dot_flux_tilde_e_nue,
    interior_normal_dot_flux_tilde_e_bar_nue,
    interior_normal_dot_flux_tilde_s_nue,
    interior_normal_dot_flux_tilde_s_bar_nue, exterior_tilde_e_nue,
    exterior_tilde_e_bar_nue, exterior_tilde_s_nue, exterior_tilde_s_bar_nue,
    exterior_normal_dot_flux_tilde_e_nue,
    exterior_normal_dot_flux_tilde_e_bar_nue,
    exterior_normal_dot_flux_tilde_s_nue,
    exterior_normal_dot_flux_tilde_s_bar_nue, use_strong_form):
    if use_strong_form:
        return (-0.5 * (interior_normal_dot_flux_tilde_s_nue +
                        exterior_normal_dot_flux_tilde_s_nue) -
                0.5 * _max_abs_char_speed *
                (exterior_tilde_s_nue - interior_tilde_s_nue))
    else:
        return (0.5 * (interior_normal_dot_flux_tilde_s_nue -
                       exterior_normal_dot_flux_tilde_s_nue) -
                0.5 * _max_abs_char_speed *
                (exterior_tilde_s_nue - interior_tilde_s_nue))


def dg_boundary_terms_tilde_s_bar_nue(
    interior_tilde_e_nue, interior_tilde_e_bar_nue, interior_tilde_s_nue,
    interior_tilde_s_bar_nue, interior_normal_dot_flux_tilde_e_nue,
    interior_normal_dot_flux_tilde_e_bar_nue,
    interior_normal_dot_flux_tilde_s_nue,
    interior_normal_dot_flux_tilde_s_bar_nue, exterior_tilde_e_nue,
    exterior_tilde_e_bar_nue, exterior_tilde_s_nue, exterior_tilde_s_bar_nue,
    exterior_normal_dot_flux_tilde_e_nue,
    exterior_normal_dot_flux_tilde_e_bar_nue,
    exterior_normal_dot_flux_tilde_s_nue,
    exterior_normal_dot_flux_tilde_s_bar_nue, use_strong_form):
    if use_strong_form:
        return (-0.5 * (interior_normal_dot_flux_tilde_s_bar_nue +
                        exterior_normal_dot_flux_tilde_s_bar_nue) -
                0.5 * _max_abs_char_speed *
                (exterior_tilde_s_bar_nue - interior_tilde_s_bar_nue))
    else:
        return (0.5 * (interior_normal_dot_flux_tilde_s_bar_nue -
                       exterior_normal_dot_flux_tilde_s_bar_nue) -
                0.5 * _max_abs_char_speed *
                (exterior_tilde_s_bar_nue - interior_tilde_s_bar_nue))
