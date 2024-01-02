# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

_max_abs_char_speed = 1.0


def dg_package_data(
    tilde_e_nue,
    tilde_e_bar_nue,
    tilde_s_nue,
    tilde_s_bar_nue,
    flux_tilde_e_nue,
    flux_tilde_e_bar_nue,
    flux_tilde_s_nue,
    flux_tilde_s_bar_nue,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return (
        tilde_e_nue,
        tilde_e_bar_nue,
        tilde_s_nue,
        tilde_s_bar_nue,
        np.asarray(np.einsum("i,i", normal_covector, flux_tilde_e_nue)),
        np.asarray(np.einsum("i,i", normal_covector, flux_tilde_e_bar_nue)),
        np.asarray(np.einsum("i,ij->j", normal_covector, flux_tilde_s_nue)),
        np.asarray(np.einsum("i,ij->j", normal_covector, flux_tilde_s_bar_nue)),
    )


def dg_boundary_terms(
    interior_tilde_e_nue,
    interior_tilde_e_bar_nue,
    interior_tilde_s_nue,
    interior_tilde_s_bar_nue,
    interior_normal_dot_flux_tilde_e_nue,
    interior_normal_dot_flux_tilde_e_bar_nue,
    interior_normal_dot_flux_tilde_s_nue,
    interior_normal_dot_flux_tilde_s_bar_nue,
    exterior_tilde_e_nue,
    exterior_tilde_e_bar_nue,
    exterior_tilde_s_nue,
    exterior_tilde_s_bar_nue,
    exterior_normal_dot_flux_tilde_e_nue,
    exterior_normal_dot_flux_tilde_e_bar_nue,
    exterior_normal_dot_flux_tilde_s_nue,
    exterior_normal_dot_flux_tilde_s_bar_nue,
    use_strong_form,
):
    sign_for_form = 1.0 if use_strong_form else -1.0

    # Use scope and locals() to get arguments into the eval context below
    scope = locals()

    def impl(var_name):
        return np.asarray(
            (
                -0.5
                * (
                    sign_for_form
                    * eval("interior_normal_dot_flux_" + var_name, scope)
                    + eval("exterior_normal_dot_flux_" + var_name, scope)
                )
                - 0.5
                * _max_abs_char_speed
                * (
                    eval("exterior_" + var_name, scope)
                    - eval("interior_" + var_name, scope)
                )
            )
        )

    return (
        impl("tilde_e_nue"),
        impl("tilde_e_bar_nue"),
        impl("tilde_s_nue"),
        impl("tilde_s_bar_nue"),
    )
