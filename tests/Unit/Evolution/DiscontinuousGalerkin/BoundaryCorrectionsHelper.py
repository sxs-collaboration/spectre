# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data_var1(var1, var2, flux_var1, flux_var2, normal_covector,
                         mesh_velocity, normal_dot_mesh_velocity,
                         volume_double):
    return var1


def dg_package_data_var1_normal_dot_flux(var1, var2, flux_var1, flux_var2,
                                         normal_covector, mesh_velocity,
                                         normal_dot_mesh_velocity,
                                         volume_double):
    return np.einsum("i,i->", flux_var1, normal_covector)


def dg_package_data_var2(var1, var2, flux_var1, flux_var2, normal_covector,
                         mesh_velocity, normal_dot_mesh_velocity,
                         volume_double):
    return var2


def dg_package_data_var2_normal_dot_flux(var1, var2, flux_var1, flux_var2,
                                         normal_covector, mesh_velocity,
                                         normal_dot_mesh_velocity,
                                         volume_double):
    return np.einsum("ij,j->i", flux_var2, normal_covector)


def dg_package_data_abs_char_speed(var1, var2, flux_var1, flux_var2,
                                   normal_covector, mesh_velocity,
                                   normal_dot_mesh_velocity, volume_double):
    if not isinstance(volume_double, float):
        volume_double = volume_double[0]
    if normal_dot_mesh_velocity is None:
        return np.abs(volume_double * var1)
    else:
        return np.abs(volume_double * var1 - normal_dot_mesh_velocity)


def dg_boundary_terms_var1(var1_int, normal_dot_flux_var1_int, var2_int,
                           normal_dot_flux_var2_int, abs_char_speed_int,
                           var1_ext, normal_dot_flux_var1_ext, var2_ext,
                           normal_dot_flux_var2_ext, abs_char_speed_ext,
                           use_strong_form):
    if use_strong_form:
        return (-0.5 * (normal_dot_flux_var1_int + normal_dot_flux_var1_ext) -
                0.5 * np.maximum(abs_char_speed_int, abs_char_speed_ext) *
                (var1_ext - var1_int))
    else:
        return (0.5 * (normal_dot_flux_var1_int - normal_dot_flux_var1_ext) -
                0.5 * np.maximum(abs_char_speed_int, abs_char_speed_ext) *
                (var1_ext - var1_int))


def dg_boundary_terms_var2(var1_int, normal_dot_flux_var1_int, var2_int,
                           normal_dot_flux_var2_int, abs_char_speed_int,
                           var1_ext, normal_dot_flux_var1_ext, var2_ext,
                           normal_dot_flux_var2_ext, abs_char_speed_ext,
                           use_strong_form):
    if use_strong_form:
        return (-0.5 * (normal_dot_flux_var2_int + normal_dot_flux_var2_ext) -
                0.5 * np.maximum(abs_char_speed_int, abs_char_speed_ext) *
                (var2_ext - var2_int))
    else:
        return (0.5 * (normal_dot_flux_var2_int - normal_dot_flux_var2_ext) -
                0.5 * np.maximum(abs_char_speed_int, abs_char_speed_ext) *
                (var2_ext - var2_int))
