# Distributed under the MIT License.
# See LICENSE.txt for details.

import math


def ideal_fluid_pressure_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    adiabatic_index,
    minimum_temperature,
):
    return (
        rest_mass_density * specific_internal_energy * (adiabatic_index - 1.0)
    )


def ideal_fluid_temperature_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    adiabatic_index,
    minimum_temperature,
):
    return (adiabatic_index - 1.0) * specific_internal_energy


def ideal_fluid_rel_pressure_from_density_and_enthalpy(
    rest_mass_density, specific_enthalpy, adiabatic_index, minimum_temperature
):
    return (
        rest_mass_density
        * (specific_enthalpy - 1.0)
        * (adiabatic_index - 1.0)
        / adiabatic_index
    )


def ideal_fluid_newt_pressure_from_density_and_enthalpy(
    rest_mass_density, specific_enthalpy, adiabatic_index, minimum_temperature
):
    return (
        rest_mass_density
        * specific_enthalpy
        * (adiabatic_index - 1.0)
        / adiabatic_index
    )


def ideal_fluid_rel_specific_enthalpy_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    adiabatic_index,
    minimum_temperature,
):
    return 1.0 + adiabatic_index * specific_internal_energy


def ideal_fluid_newt_specific_enthalpy_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    adiabatic_index,
    minimum_temperature,
):
    return adiabatic_index * specific_internal_energy


def ideal_fluid_specific_entropy_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    adiabatic_index,
    minimum_temperature,
):
    return math.log(
        max(
            specific_internal_energy,
            minimum_temperature / (adiabatic_index - 1.0),
        )
        / rest_mass_density ** (adiabatic_index - 1.0)
    ) / (adiabatic_index - 1.0)


def ideal_fluid_specific_entropy_from_density_and_temperature(
    rest_mass_density,
    temperature,
    adiabatic_index,
    minimum_temperature,
):
    return math.log(
        max(temperature, minimum_temperature)
        / (
            (adiabatic_index - 1.0)
            * rest_mass_density ** (adiabatic_index - 1.0)
        )
    ) / (adiabatic_index - 1.0)


def ideal_fluid_specific_internal_energy_from_density_and_pressure(
    rest_mass_density, pressure, adiabatic_index, minimum_temperature
):
    return pressure / (adiabatic_index - 1.0) / rest_mass_density


def ideal_fluid_chi_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    adiabatic_index,
    minimum_temperature,
):
    return specific_internal_energy * (adiabatic_index - 1.0)


def ideal_fluid_kappa_times_p_over_rho_squared_from_density_and_energy(
    rest_mass_density,
    specific_internal_energy,
    adiabatic_index,
    minimum_temperature,
):
    return specific_internal_energy * (adiabatic_index - 1.0) ** 2
