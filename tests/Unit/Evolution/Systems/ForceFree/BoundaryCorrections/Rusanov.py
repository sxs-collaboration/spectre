# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    tilde_e,
    tilde_b,
    tilde_psi,
    tilde_phi,
    tilde_q,
    flux_tilde_e,
    flux_tilde_b,
    flux_tilde_psi,
    flux_tilde_phi,
    flux_tilde_q,
    lapse,
    shift,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
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
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        np.asarray(np.einsum("ij,i->j", flux_tilde_e, normal_covector)),
        np.asarray(np.einsum("ij,i->j", flux_tilde_b, normal_covector)),
        np.asarray(np.dot(flux_tilde_psi, normal_covector)),
        np.asarray(np.dot(flux_tilde_phi, normal_covector)),
        np.asarray(np.dot(flux_tilde_q, normal_covector)),
        np.asarray(np.maximum(compute_char(1.0), compute_char(-1.0))),
    )


def dg_boundary_terms(
    interior_tilde_e,
    interior_tilde_b,
    interior_tilde_psi,
    interior_tilde_phi,
    interior_tilde_q,
    interior_normal_dot_flux_tilde_e,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_psi,
    interior_normal_dot_flux_tilde_phi,
    interior_normal_dot_flux_tilde_q,
    interior_abs_char_speed,
    exterior_tilde_e,
    exterior_tilde_b,
    exterior_tilde_psi,
    exterior_tilde_phi,
    exterior_tilde_q,
    exterior_normal_dot_flux_tilde_e,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_psi,
    exterior_normal_dot_flux_tilde_phi,
    exterior_normal_dot_flux_tilde_q,
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
        impl("tilde_e"),
        impl("tilde_b"),
        impl("tilde_psi"),
        impl("tilde_phi"),
        impl("tilde_q"),
    )
