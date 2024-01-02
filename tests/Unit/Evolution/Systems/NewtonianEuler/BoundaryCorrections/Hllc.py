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

    if use_polytropic_fluid:
        polytropic_constant = 1.0e-3
        polytropic_exponent = 2.0
        pressure = polytropic_constant * pow(mass_density, polytropic_exponent)
    else:
        adiabatic_index = 1.3
        pressure = (
            mass_density * specific_internal_energy * (adiabatic_index - 1.0)
        )

    return (
        mass_density,
        momentum_density,
        energy_density,
        np.asarray(pressure),
        np.asarray(np.einsum("i,i", normal_covector, flux_mass_density)),
        np.asarray(
            np.einsum("i,ij->j", normal_covector, flux_momentum_density)
        ),
        np.asarray(np.einsum("i,i", normal_covector, flux_energy_density)),
        normal_covector,
        np.asarray(
            np.einsum("i,i", normal_covector, velocity)
            if normal_dot_mesh_velocity is None
            else (
                np.einsum("i,i", normal_covector, velocity)
                - normal_dot_mesh_velocity
            )
        ),
        np.asarray(
            velocity_dot_normal + sound_speed
            if normal_dot_mesh_velocity is None
            else velocity_dot_normal + sound_speed - normal_dot_mesh_velocity
        ),
        np.asarray(
            velocity_dot_normal - sound_speed
            if normal_dot_mesh_velocity is None
            else velocity_dot_normal - sound_speed - normal_dot_mesh_velocity
        ),
    )


def dg_boundary_terms_mass_density(
    interior_mass_density,
    interior_momentum_density,
    interior_energy_density,
    interior_pressure,
    interior_normal_dot_flux_mass_density,
    interior_normal_dot_flux_momentum_density,
    interior_normal_dot_flux_energy_density,
    interior_interface_unit_normal,
    interior_normal_dot_velocity,
    interior_largest_outgoing_char_speed,
    interior_largest_ingoing_char_speed,
    exterior_mass_density,
    exterior_momentum_density,
    exterior_energy_density,
    exterior_pressure,
    exterior_normal_dot_flux_mass_density,
    exterior_normal_dot_flux_momentum_density,
    exterior_normal_dot_flux_energy_density,
    exterior_interface_unit_normal,
    exterior_normal_dot_velocity,
    exterior_largest_outgoing_char_speed,
    exterior_largest_ingoing_char_speed,
    use_strong_form,
):
    lambda_max = np.maximum(
        0.0,
        np.maximum(
            interior_largest_outgoing_char_speed,
            -exterior_largest_ingoing_char_speed,
        ),
    )
    lambda_min = np.minimum(
        0.0,
        np.minimum(
            interior_largest_ingoing_char_speed,
            -exterior_largest_outgoing_char_speed,
        ),
    )

    lambda_star = (
        exterior_pressure
        - interior_pressure
        + interior_mass_density
        * interior_normal_dot_velocity
        * (lambda_min - interior_normal_dot_velocity)
        + exterior_mass_density
        * exterior_normal_dot_velocity
        * (lambda_max + exterior_normal_dot_velocity)
    ) / (
        interior_mass_density * (lambda_min - interior_normal_dot_velocity)
        - exterior_mass_density * (lambda_max + exterior_normal_dot_velocity)
    )

    prefactor_int = (lambda_min - interior_normal_dot_velocity) / (
        lambda_min - lambda_star
    )
    prefactor_ext = (lambda_max + exterior_normal_dot_velocity) / (
        lambda_max - lambda_star
    )

    if use_strong_form:
        return np.where(
            lambda_star >= 0.0,
            lambda_min * (prefactor_int - 1.0) * interior_mass_density,
            -interior_normal_dot_flux_mass_density
            - exterior_normal_dot_flux_mass_density
            + lambda_max * (prefactor_ext - 1.0) * exterior_mass_density,
        )
    else:
        return np.where(
            lambda_star >= 0.0,
            interior_normal_dot_flux_mass_density
            + lambda_min * (prefactor_int - 1.0) * interior_mass_density,
            -exterior_normal_dot_flux_mass_density
            + lambda_max * (prefactor_ext - 1.0) * exterior_mass_density,
        )


def dg_boundary_terms_momentum_density(
    interior_mass_density,
    interior_momentum_density,
    interior_energy_density,
    interior_pressure,
    interior_normal_dot_flux_mass_density,
    interior_normal_dot_flux_momentum_density,
    interior_normal_dot_flux_energy_density,
    interior_interface_unit_normal,
    interior_normal_dot_velocity,
    interior_largest_outgoing_char_speed,
    interior_largest_ingoing_char_speed,
    exterior_mass_density,
    exterior_momentum_density,
    exterior_energy_density,
    exterior_pressure,
    exterior_normal_dot_flux_mass_density,
    exterior_normal_dot_flux_momentum_density,
    exterior_normal_dot_flux_energy_density,
    exterior_interface_unit_normal,
    exterior_normal_dot_velocity,
    exterior_largest_outgoing_char_speed,
    exterior_largest_ingoing_char_speed,
    use_strong_form,
):
    lambda_max = np.maximum(
        0.0,
        np.maximum(
            interior_largest_outgoing_char_speed,
            -exterior_largest_ingoing_char_speed,
        ),
    )
    lambda_min = np.minimum(
        0.0,
        np.minimum(
            interior_largest_ingoing_char_speed,
            -exterior_largest_outgoing_char_speed,
        ),
    )

    lambda_star = (
        exterior_pressure
        - interior_pressure
        + interior_mass_density
        * interior_normal_dot_velocity
        * (lambda_min - interior_normal_dot_velocity)
        + exterior_mass_density
        * exterior_normal_dot_velocity
        * (lambda_max + exterior_normal_dot_velocity)
    ) / (
        interior_mass_density * (lambda_min - interior_normal_dot_velocity)
        - exterior_mass_density * (lambda_max + exterior_normal_dot_velocity)
    )

    prefactor_int = (lambda_min - interior_normal_dot_velocity) / (
        lambda_min - lambda_star
    )
    prefactor_ext = (lambda_max + exterior_normal_dot_velocity) / (
        lambda_max - lambda_star
    )

    if use_strong_form:
        return np.where(
            lambda_star >= 0.0,
            lambda_min
            * (
                interior_mass_density
                * prefactor_int
                * (lambda_star - interior_normal_dot_velocity)
                * interior_interface_unit_normal
                + interior_momentum_density * (prefactor_int - 1.0)
            ),
            -interior_normal_dot_flux_momentum_density
            - exterior_normal_dot_flux_momentum_density
            + lambda_max
            * (
                exterior_mass_density
                * prefactor_ext
                * (lambda_star + exterior_normal_dot_velocity)
                * (-exterior_interface_unit_normal)
                + exterior_momentum_density * (prefactor_ext - 1.0)
            ),
        )
    else:
        return np.where(
            lambda_star >= 0.0,
            interior_normal_dot_flux_momentum_density
            + lambda_min
            * (
                interior_mass_density
                * prefactor_int
                * (lambda_star - interior_normal_dot_velocity)
                * interior_interface_unit_normal
                + interior_momentum_density * (prefactor_int - 1.0)
            ),
            -exterior_normal_dot_flux_momentum_density
            + lambda_max
            * (
                exterior_mass_density
                * prefactor_ext
                * (lambda_star + exterior_normal_dot_velocity)
                * (-exterior_interface_unit_normal)
                + exterior_momentum_density * (prefactor_ext - 1.0)
            ),
        )


def dg_boundary_terms_energy_density(
    interior_mass_density,
    interior_momentum_density,
    interior_energy_density,
    interior_pressure,
    interior_normal_dot_flux_mass_density,
    interior_normal_dot_flux_momentum_density,
    interior_normal_dot_flux_energy_density,
    interior_interface_unit_normal,
    interior_normal_dot_velocity,
    interior_largest_outgoing_char_speed,
    interior_largest_ingoing_char_speed,
    exterior_mass_density,
    exterior_momentum_density,
    exterior_energy_density,
    exterior_pressure,
    exterior_normal_dot_flux_mass_density,
    exterior_normal_dot_flux_momentum_density,
    exterior_normal_dot_flux_energy_density,
    exterior_interface_unit_normal,
    exterior_normal_dot_velocity,
    exterior_largest_outgoing_char_speed,
    exterior_largest_ingoing_char_speed,
    use_strong_form,
):
    lambda_max = np.maximum(
        0.0,
        np.maximum(
            interior_largest_outgoing_char_speed,
            -exterior_largest_ingoing_char_speed,
        ),
    )
    lambda_min = np.minimum(
        0.0,
        np.minimum(
            interior_largest_ingoing_char_speed,
            -exterior_largest_outgoing_char_speed,
        ),
    )

    lambda_star = (
        exterior_pressure
        - interior_pressure
        + interior_mass_density
        * interior_normal_dot_velocity
        * (lambda_min - interior_normal_dot_velocity)
        + exterior_mass_density
        * exterior_normal_dot_velocity
        * (lambda_max + exterior_normal_dot_velocity)
    ) / (
        interior_mass_density * (lambda_min - interior_normal_dot_velocity)
        - exterior_mass_density * (lambda_max + exterior_normal_dot_velocity)
    )

    prefactor_int = (lambda_min - interior_normal_dot_velocity) / (
        lambda_min - lambda_star
    )
    prefactor_ext = (lambda_max + exterior_normal_dot_velocity) / (
        lambda_max - lambda_star
    )

    if use_strong_form:
        return np.where(
            lambda_star >= 0.0,
            lambda_min
            * (
                (prefactor_int - 1.0)
                * (interior_energy_density + interior_pressure)
                + prefactor_int
                * interior_mass_density
                * lambda_star
                * (lambda_star - interior_normal_dot_velocity)
            ),
            -interior_normal_dot_flux_energy_density
            - exterior_normal_dot_flux_energy_density
            + lambda_max
            * (
                (prefactor_ext - 1.0)
                * (exterior_energy_density + exterior_pressure)
                + prefactor_ext
                * exterior_mass_density
                * lambda_star
                * (lambda_star + exterior_normal_dot_velocity)
            ),
        )
    else:
        return np.where(
            lambda_star >= 0.0,
            interior_normal_dot_flux_energy_density
            + lambda_min
            * (
                prefactor_int
                * (
                    interior_energy_density
                    + interior_pressure
                    * (lambda_star - interior_normal_dot_velocity)
                    / (lambda_min - interior_normal_dot_velocity)
                    + interior_mass_density
                    * lambda_star
                    * (lambda_star - interior_normal_dot_velocity)
                )
                - interior_energy_density
            ),
            -exterior_normal_dot_flux_energy_density
            + lambda_max
            * (
                prefactor_ext
                * (
                    exterior_energy_density
                    + exterior_pressure
                    * (lambda_star + exterior_normal_dot_velocity)
                    / (lambda_max + exterior_normal_dot_velocity)
                    + exterior_mass_density
                    * lambda_star
                    * (lambda_star + exterior_normal_dot_velocity)
                )
                - exterior_energy_density
            ),
        )


def dg_boundary_terms(
    interior_mass_density,
    interior_momentum_density,
    interior_energy_density,
    interior_pressure,
    interior_normal_dot_flux_mass_density,
    interior_normal_dot_flux_momentum_density,
    interior_normal_dot_flux_energy_density,
    interior_interface_unit_normal,
    interior_normal_dot_velocity,
    interior_largest_outgoing_char_speed,
    interior_largest_ingoing_char_speed,
    exterior_mass_density,
    exterior_momentum_density,
    exterior_energy_density,
    exterior_pressure,
    exterior_normal_dot_flux_mass_density,
    exterior_normal_dot_flux_momentum_density,
    exterior_normal_dot_flux_energy_density,
    exterior_interface_unit_normal,
    exterior_normal_dot_velocity,
    exterior_largest_outgoing_char_speed,
    exterior_largest_ingoing_char_speed,
    use_strong_form,
):
    lambda_max = np.maximum(
        0.0,
        np.maximum(
            interior_largest_outgoing_char_speed,
            -exterior_largest_ingoing_char_speed,
        ),
    )
    lambda_min = np.minimum(
        0.0,
        np.minimum(
            interior_largest_ingoing_char_speed,
            -exterior_largest_outgoing_char_speed,
        ),
    )

    lambda_star = (
        exterior_pressure
        - interior_pressure
        + interior_mass_density
        * interior_normal_dot_velocity
        * (lambda_min - interior_normal_dot_velocity)
        + exterior_mass_density
        * exterior_normal_dot_velocity
        * (lambda_max + exterior_normal_dot_velocity)
    ) / (
        interior_mass_density * (lambda_min - interior_normal_dot_velocity)
        - exterior_mass_density * (lambda_max + exterior_normal_dot_velocity)
    )

    prefactor_int = (lambda_min - interior_normal_dot_velocity) / (
        lambda_min - lambda_star
    )
    prefactor_ext = (lambda_max + exterior_normal_dot_velocity) / (
        lambda_max - lambda_star
    )

    # Use scope and locals() to get arguments into the eval context below
    scope = locals()

    def mass_density_impl():
        var_name = "mass_density"
        if use_strong_form:
            return np.where(
                lambda_star >= 0.0,
                lambda_min
                * (prefactor_int - 1.0)
                * eval("interior_" + var_name, scope),
                -eval("interior_normal_dot_flux_" + var_name, scope)
                - eval("exterior_normal_dot_flux_" + var_name, scope)
                + lambda_max
                * (prefactor_ext - 1.0)
                * eval("exterior_" + var_name, scope),
            )
        else:
            return np.where(
                lambda_star >= 0.0,
                eval("interior_normal_dot_flux_" + var_name, scope)
                + lambda_min
                * (prefactor_int - 1.0)
                * eval("interior_" + var_name, scope),
                -eval("exterior_normal_dot_flux_" + var_name, scope)
                + lambda_max
                * (prefactor_ext - 1.0)
                * eval("exterior_" + var_name, scope),
            )

    def momentum_density_impl():
        if use_strong_form:
            return np.where(
                lambda_star >= 0.0,
                lambda_min
                * (
                    interior_mass_density
                    * prefactor_int
                    * (lambda_star - interior_normal_dot_velocity)
                    * interior_interface_unit_normal
                    + interior_momentum_density * (prefactor_int - 1.0)
                ),
                -interior_normal_dot_flux_momentum_density
                - exterior_normal_dot_flux_momentum_density
                + lambda_max
                * (
                    exterior_mass_density
                    * prefactor_ext
                    * (lambda_star + exterior_normal_dot_velocity)
                    * (-exterior_interface_unit_normal)
                    + exterior_momentum_density * (prefactor_ext - 1.0)
                ),
            )
        else:
            return np.where(
                lambda_star >= 0.0,
                interior_normal_dot_flux_momentum_density
                + lambda_min
                * (
                    interior_mass_density
                    * prefactor_int
                    * (lambda_star - interior_normal_dot_velocity)
                    * interior_interface_unit_normal
                    + interior_momentum_density * (prefactor_int - 1.0)
                ),
                -exterior_normal_dot_flux_momentum_density
                + lambda_max
                * (
                    exterior_mass_density
                    * prefactor_ext
                    * (lambda_star + exterior_normal_dot_velocity)
                    * (-exterior_interface_unit_normal)
                    + exterior_momentum_density * (prefactor_ext - 1.0)
                ),
            )

    def energy_density_impl():
        if use_strong_form:
            return np.where(
                lambda_star >= 0.0,
                lambda_min
                * (
                    (prefactor_int - 1.0)
                    * (interior_energy_density + interior_pressure)
                    + prefactor_int
                    * interior_mass_density
                    * lambda_star
                    * (lambda_star - interior_normal_dot_velocity)
                ),
                -interior_normal_dot_flux_energy_density
                - exterior_normal_dot_flux_energy_density
                + lambda_max
                * (
                    (prefactor_ext - 1.0)
                    * (exterior_energy_density + exterior_pressure)
                    + prefactor_ext
                    * exterior_mass_density
                    * lambda_star
                    * (lambda_star + exterior_normal_dot_velocity)
                ),
            )
        else:
            return np.where(
                lambda_star >= 0.0,
                interior_normal_dot_flux_energy_density
                + lambda_min
                * (
                    prefactor_int
                    * (
                        interior_energy_density
                        + interior_pressure
                        * (lambda_star - interior_normal_dot_velocity)
                        / (lambda_min - interior_normal_dot_velocity)
                        + interior_mass_density
                        * lambda_star
                        * (lambda_star - interior_normal_dot_velocity)
                    )
                    - interior_energy_density
                ),
                -exterior_normal_dot_flux_energy_density
                + lambda_max
                * (
                    prefactor_ext
                    * (
                        exterior_energy_density
                        + exterior_pressure
                        * (lambda_star + exterior_normal_dot_velocity)
                        / (lambda_max + exterior_normal_dot_velocity)
                        + exterior_mass_density
                        * lambda_star
                        * (lambda_star + exterior_normal_dot_velocity)
                    )
                    - exterior_energy_density
                ),
            )

    return (
        mass_density_impl(),
        momentum_density_impl(),
        energy_density_impl(),
    )
