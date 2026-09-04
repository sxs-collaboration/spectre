# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    var1,
    var2,
    flux_var1,
    flux_var2,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    volume_double,
):
    if not isinstance(volume_double, float):
        volume_double = volume_double[0]
    if normal_dot_mesh_velocity is None:
        abs_char_speed = np.abs(volume_double * var1)
    else:
        abs_char_speed = np.abs(volume_double * var1 - normal_dot_mesh_velocity)
    return (
        var1,
        np.asarray(np.einsum("i,i->", flux_var1, normal_covector)),
        var2,
        np.asarray(np.einsum("ij,j->i", flux_var2, normal_covector)),
        np.asarray(abs_char_speed),
    )


def dg_package_data_curved(
    var1,
    var2,
    flux_var1,
    flux_var2,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    volume_double,
):
    if not isinstance(volume_double, float):
        volume_double = volume_double[0]
    if normal_dot_mesh_velocity is None:
        abs_char_speed = np.abs(volume_double * var1)
    else:
        abs_char_speed = np.abs(volume_double * var1 - normal_dot_mesh_velocity)
    return (
        var1,
        np.asarray(np.einsum("i,i->", flux_var1, normal_covector)),
        var2,
        np.asarray(
            np.einsum("ij,j->i", flux_var2, normal_covector) + normal_vector
        ),
        np.asarray(abs_char_speed),
    )


def dg_boundary_terms(
    var1_int,
    normal_dot_flux_var1_int,
    var2_int,
    normal_dot_flux_var2_int,
    abs_char_speed_int,
    var1_ext,
    normal_dot_flux_var1_ext,
    var2_ext,
    normal_dot_flux_var2_ext,
    abs_char_speed_ext,
    use_strong_form,
    volume_double,
):
    if not isinstance(volume_double, float):
        volume_double = volume_double[0]
    assert volume_double == 2.3
    if use_strong_form:
        return (
            np.asarray(
                -0.5 * (normal_dot_flux_var1_int + normal_dot_flux_var1_ext)
                - 0.5
                * np.maximum(abs_char_speed_int, abs_char_speed_ext)
                * (var1_ext - var1_int)
            ),
            np.asarray(
                -0.5 * (normal_dot_flux_var2_int + normal_dot_flux_var2_ext)
                - 0.5
                * np.maximum(abs_char_speed_int, abs_char_speed_ext)
                * (var2_ext - var2_int)
            ),
        )
    else:
        return (
            np.asarray(
                0.5 * (normal_dot_flux_var1_int - normal_dot_flux_var1_ext)
                - 0.5
                * np.maximum(abs_char_speed_int, abs_char_speed_ext)
                * (var1_ext - var1_int)
            ),
            np.asarray(
                0.5 * (normal_dot_flux_var2_int - normal_dot_flux_var2_ext)
                - 0.5
                * np.maximum(abs_char_speed_int, abs_char_speed_ext)
                * (var2_ext - var2_int)
            ),
        )


def dg_package_data_aux_system(
    var1,
    var2,
    aux_var,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    volume_double,
    extra_arg,
):
    if not isinstance(volume_double, float):
        volume_double = volume_double[0]
    assert volume_double == 2.3
    assert extra_arg == 1.25
    return (
        var1,
        np.asarray(np.einsum("i,i", normal_covector, aux_var)),
        var1 * normal_covector,
    )


def dg_boundary_terms_aux_system(
    var1_int,
    normal_dot_aux_int,
    var1_times_normal_int,
    var1_ext,
    normal_dot_aux_ext,
    var1_times_normal_ext,
    use_strong_form,
    aux_boundary_double,
    extra_arg,
):
    if not isinstance(aux_boundary_double, float):
        aux_boundary_double = aux_boundary_double[0]
    assert aux_boundary_double == 3.7
    assert extra_arg == 1.25
    return (
        np.asarray(-0.5 * (normal_dot_aux_int + normal_dot_aux_ext)),
        -0.5 * (var1_times_normal_int + var1_times_normal_ext),
    )


def dg_auxiliary_package_data_aux_system(
    var1,
    var2,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    volume_double,
    extra_arg,
):
    if not isinstance(volume_double, float):
        volume_double = volume_double[0]
    assert volume_double == 2.3
    assert extra_arg == 1.25
    return (var1, var1 * normal_covector)


def dg_auxiliary_boundary_terms_aux_system(
    var1_int,
    var1_times_normal_int,
    var1_ext,
    var1_times_normal_ext,
    use_strong_form,
    aux_boundary_double,
    extra_arg,
):
    if not isinstance(aux_boundary_double, float):
        aux_boundary_double = aux_boundary_double[0]
    assert aux_boundary_double == 3.7
    assert extra_arg == 1.25
    return (0.5 * (var1_times_normal_int + var1_times_normal_ext),)
