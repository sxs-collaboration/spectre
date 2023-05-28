# Distributed under the MIT License.
# See LICENSE.txt for details.

from PolytropicFluid import (
    polytropic_pressure_from_density,
    polytropic_specific_internal_energy_from_density,
)


def hybrid_polytrope_pressure_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    p_c = polytropic_pressure_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return p_c + rest_mass_density * (specific_internal_energy - eps_c) * (
        thermal_adiabatic_index - 1.0
    )


def hybrid_polytrope_rel_pressure_from_density_and_enthalpy(
    rest_mass_density,
    specific_enthalpy,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    p_c = polytropic_pressure_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return p_c / thermal_adiabatic_index + (
        rest_mass_density
        * (specific_enthalpy - 1.0 - eps_c)
        * (thermal_adiabatic_index - 1.0)
        / thermal_adiabatic_index
    )


def hybrid_polytrope_temperature_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return (thermal_adiabatic_index - 1.0) * (specific_internal_energy - eps_c)


def hybrid_polytrope_newt_pressure_from_density_and_enthalpy(
    rest_mass_density,
    specific_enthalpy,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    p_c = polytropic_pressure_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return p_c / thermal_adiabatic_index + (
        rest_mass_density
        * (specific_enthalpy - eps_c)
        * (thermal_adiabatic_index - 1.0)
        / thermal_adiabatic_index
    )


def hybrid_polytrope_rel_specific_enthalpy_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    p_c = polytropic_pressure_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return (
        1.0
        + eps_c
        + p_c / rest_mass_density
        + thermal_adiabatic_index * (specific_internal_energy - eps_c)
    )


def hybrid_polytrope_newt_specific_enthalpy_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    p_c = polytropic_pressure_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return (
        eps_c
        + p_c / rest_mass_density
        + thermal_adiabatic_index * (specific_internal_energy - eps_c)
    )


def hybrid_polytrope_specific_internal_energy_from_density_and_pressure(
    rest_mass_density,
    pressure,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    p_c = polytropic_pressure_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return (
        eps_c
        + (pressure - p_c) / (thermal_adiabatic_index - 1.0) / rest_mass_density
    )


def hybrid_polytrope_chi_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    p_c = polytropic_pressure_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return polytropic_exponent * p_c / rest_mass_density + (
        specific_internal_energy - eps_c - p_c / rest_mass_density
    ) * (thermal_adiabatic_index - 1.0)


def hybrid_polytrope_kappa_times_p_over_rho_squared_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    polytropic_constant,
    polytropic_exponent,
    thermal_adiabatic_index,
):
    p_c = polytropic_pressure_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    eps_c = polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent
    )
    return (thermal_adiabatic_index - 1.0) * p_c / rest_mass_density + (
        specific_internal_energy - eps_c
    ) * (thermal_adiabatic_index - 1.0) ** 2
