# Distributed under the MIT License.
# See LICENSE.txt for details.

import Evolution.Systems.RelativisticEuler.Valencia.TestFunctions as valencia
import numpy as np


def dg_package_data(
    tilde_d,
    tilde_tau,
    tilde_s,
    flux_tilde_d,
    flux_tilde_tau,
    flux_tilde_s,
    lapse,
    shift,
    spatial_metric,
    rest_mass_density,
    specific_internal_energy,
    specific_enthalpy,
    spatial_velocity,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    use_polytropic_fluid,
):
    spatial_velocity_squared = np.einsum(
        "ij,i,j", spatial_metric, spatial_velocity, spatial_velocity
    )

    # Note that the relativistic sound speed squared has a 1/enthalpy
    if use_polytropic_fluid:
        polytropic_constant = 1.0e-3
        polytropic_exponent = 2.0
        sound_speed_squared = (
            polytropic_constant
            * polytropic_exponent
            * pow(rest_mass_density, polytropic_exponent - 1.0)
            / specific_enthalpy
        )
    else:
        adiabatic_index = 1.3
        chi = specific_internal_energy * (adiabatic_index - 1.0)
        kappa_times_p_over_rho_squared = (
            adiabatic_index - 1.0
        ) ** 2 * specific_internal_energy
        sound_speed_squared = (
            chi + kappa_times_p_over_rho_squared
        ) / specific_enthalpy

    char_speeds = valencia.characteristic_speeds(
        lapse,
        shift,
        spatial_velocity,
        spatial_velocity_squared,
        sound_speed_squared,
        normal_covector,
    )

    return (
        tilde_d,
        tilde_tau,
        tilde_s,
        np.asarray(np.dot(flux_tilde_d, normal_covector)),
        np.asarray(np.dot(flux_tilde_tau, normal_covector)),
        np.einsum("ij,i->j", flux_tilde_s, normal_covector),
        np.asarray(
            np.max(np.abs(char_speeds))
            if normal_dot_mesh_velocity is None
            else np.max(np.abs(char_speeds - normal_dot_mesh_velocity))
        ),
    )


def dg_boundary_terms(
    interior_tilde_d,
    interior_tilde_tau,
    interior_tilde_s,
    interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau,
    interior_normal_dot_flux_tilde_s,
    interior_abs_char_speed,
    exterior_tilde_d,
    exterior_tilde_tau,
    exterior_tilde_s,
    exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s,
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
        impl("tilde_tau"),
        impl("tilde_s"),
    )
