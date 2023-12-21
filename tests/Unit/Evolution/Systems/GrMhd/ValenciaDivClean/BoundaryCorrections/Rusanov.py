# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    tilde_d,
    tilde_ye,
    tilde_tau,
    tilde_s,
    tilde_b,
    tilde_phi,
    flux_tilde_d,
    flux_tilde_ye,
    flux_tilde_tau,
    flux_tilde_s,
    flux_tilde_b,
    flux_tilde_phi,
    lapse,
    shift,
    spatial_velocity_one_form,
    rest_mass_density,
    electron_fraction,
    temperature,
    spatial_velocity,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    equation_of_state,
):
    def compute_char(lapse_sign):
        return np.abs(
            np.asarray(
                (lapse_sign * lapse - np.dot(shift, normal_covector))
                if normal_dot_mesh_velocity is None
                else (
                    lapse_sign * lapse
                    - np.dot(shift, normal_covector)
                    - normal_dot_mesh_velocity
                )
            )
        )

    return (
        tilde_d,
        tilde_ye,
        tilde_tau,
        tilde_s,
        tilde_b,
        tilde_phi,
        np.asarray(np.dot(flux_tilde_d, normal_covector)),
        np.asarray(np.dot(flux_tilde_ye, normal_covector)),
        np.asarray(np.dot(flux_tilde_tau, normal_covector)),
        np.einsum("ij,i->j", flux_tilde_s, normal_covector),
        np.einsum("ij,i->j", flux_tilde_b, normal_covector),
        np.asarray(np.dot(flux_tilde_phi, normal_covector)),
        np.asarray(np.maximum(compute_char(1.0), compute_char(-1.0))),
    )


def dg_boundary_terms(
    interior_tilde_d,
    interior_tilde_ye,
    interior_tilde_tau,
    interior_tilde_s,
    interior_tilde_b,
    interior_tilde_phi,
    interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_ye,
    interior_normal_dot_flux_tilde_tau,
    interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_phi,
    interior_abs_char_speed,
    exterior_tilde_d,
    exterior_tilde_ye,
    exterior_tilde_tau,
    exterior_tilde_s,
    exterior_tilde_b,
    exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_ye,
    exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi,
    exterior_abs_char_speed,
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
                * np.maximum(interior_abs_char_speed, exterior_abs_char_speed)
                * (
                    eval("exterior_" + var_name, scope)
                    - eval("interior_" + var_name, scope)
                )
            )
        )

    return (
        impl("tilde_d"),
        impl("tilde_ye"),
        impl("tilde_tau"),
        impl("tilde_s"),
        impl("tilde_b"),
        impl("tilde_phi"),
    )
