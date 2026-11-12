# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    mass_density,
    momentum_density,
    energy_density,
    flux_mass_density,
    flux_momentum_density,
    flux_energy_density,
    velocity,
    specific_internal_energy,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    use_polytropic_fluid,
):
    velocity_dot_normal = np.einsum("i,i", normal_covector, velocity)
    if use_polytropic_fluid:
        polytropic_constant = 1.0e-3
        polytropic_exponent = 2.0
        sound_speed = np.sqrt(
            polytropic_constant
            * polytropic_exponent
            * pow(mass_density, polytropic_exponent - 1.0)
        )
    else:
        adiabatic_index = 1.3
        chi = specific_internal_energy * (adiabatic_index - 1.0)
        kappa_times_p_over_rho_squared = (
            adiabatic_index - 1.0
        ) ** 2 * specific_internal_energy
        sound_speed = np.sqrt(chi + kappa_times_p_over_rho_squared)

    return (
        mass_density,
        momentum_density,
        energy_density,
        np.asarray(np.einsum("i,i", normal_covector, flux_mass_density)),
        np.asarray(
            np.einsum("i,ij->j", normal_covector, flux_momentum_density)
        ),
        np.asarray(np.einsum("i,i", normal_covector, flux_energy_density)),
        np.asarray(
            np.maximum(
                np.abs(velocity_dot_normal - sound_speed),
                np.abs(velocity_dot_normal + sound_speed),
            )
            if normal_dot_mesh_velocity is None
            else np.maximum(
                np.abs(
                    velocity_dot_normal - sound_speed - normal_dot_mesh_velocity
                ),
                np.abs(
                    velocity_dot_normal + sound_speed - normal_dot_mesh_velocity
                ),
            )
        ),
    )


def dg_boundary_terms(
    interior_mass_density,
    interior_momentum_density,
    interior_energy_density,
    interior_normal_dot_flux_mass_density,
    interior_normal_dot_flux_momentum_density,
    interior_normal_dot_flux_energy_density,
    interior_abs_char_speed,
    exterior_mass_density,
    exterior_momentum_density,
    exterior_energy_density,
    exterior_normal_dot_flux_mass_density,
    exterior_normal_dot_flux_momentum_density,
    exterior_normal_dot_flux_energy_density,
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
        impl("mass_density"),
        impl("momentum_density"),
        impl("energy_density"),
    )
