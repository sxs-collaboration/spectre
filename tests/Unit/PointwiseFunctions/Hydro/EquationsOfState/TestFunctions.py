# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing DarkEnergyFluid.cpp
def dark_energy_fluid_pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, parameter_w):
    return parameter_w * rest_mass_density * (1.0 + specific_internal_energy)


def dark_energy_fluid_rel_pressure_from_density_and_enthalpy(
        rest_mass_density, specific_enthalpy, parameter_w):
    return (parameter_w * rest_mass_density * specific_enthalpy /
            (parameter_w + 1.0))


def dark_energy_fluid_rel_specific_enthalpy_from_density_and_energy(
        rest_mass_density, specific_internal_energy, parameter_w):
    return (parameter_w + 1.0) * (1.0 + specific_internal_energy)


def dark_energy_fluid_specific_internal_energy_from_density_and_pressure(
        rest_mass_density, pressure, parameter_w):
    return pressure / (parameter_w * rest_mass_density) - 1.0


def dark_energy_fluid_chi_from_density_and_energy(
        rest_mass_density, specific_internal_energy, parameter_w):
    return parameter_w * (1.0 + specific_internal_energy)


def dark_energy_fluid_kappa_times_p_over_rho_squared_from_density_and_energy(
        rest_mass_density, specific_internal_energy, parameter_w):
    return parameter_w**2 * (1.0 + specific_internal_energy)


# End functions for testing DarkEnergyFluid.cpp


# Functions for testing IdealFluid.cpp
def ideal_fluid_pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, adiabatic_index):
    return rest_mass_density * specific_internal_energy * (
        adiabatic_index - 1.0)


def ideal_fluid_rel_pressure_from_density_and_enthalpy(
        rest_mass_density, specific_enthalpy, adiabatic_index):
    return (rest_mass_density * (specific_enthalpy - 1.0) *
            (adiabatic_index - 1.0) / adiabatic_index)


def ideal_fluid_newt_pressure_from_density_and_enthalpy(
        rest_mass_density, specific_enthalpy, adiabatic_index):
    return (rest_mass_density * specific_enthalpy * (adiabatic_index - 1.0) /
            adiabatic_index)


def ideal_fluid_rel_specific_enthalpy_from_density_and_energy(
        rest_mass_density, specific_internal_energy, adiabatic_index):
    return 1.0 + adiabatic_index * specific_internal_energy


def ideal_fluid_newt_specific_enthalpy_from_density_and_energy(
        rest_mass_density, specific_internal_energy, adiabatic_index):
    return adiabatic_index * specific_internal_energy


def ideal_fluid_specific_internal_energy_from_density_and_pressure(
        rest_mass_density, pressure, adiabatic_index):
    return pressure / (adiabatic_index - 1.0) / rest_mass_density


def ideal_fluid_chi_from_density_and_energy(
        rest_mass_density, specific_internal_energy, adiabatic_index):
    return specific_internal_energy * (adiabatic_index - 1.0)


def ideal_fluid_kappa_times_p_over_rho_squared_from_density_and_energy(
        rest_mass_density, specific_internal_energy, adiabatic_index):
    return specific_internal_energy * (adiabatic_index - 1.0)**2


# End functions for testing IdealFluid.cpp


# Functions for testing PolytropicFluid.cpp
def polytropic_pressure_from_density(rest_mass_density, polytropic_constant,
                                     polytropic_exponent):
    return polytropic_constant * rest_mass_density**polytropic_exponent


def polytropic_rel_rest_mass_density_from_enthalpy(
        specific_enthalpy, polytropic_constant, polytropic_exponent):
    return ((polytropic_exponent - 1.0) /
            (polytropic_constant * polytropic_exponent) *
            (specific_enthalpy - 1.0))**(1.0 / (polytropic_exponent - 1.0))


def polytropic_newt_rest_mass_density_from_enthalpy(
        specific_enthalpy, polytropic_constant, polytropic_exponent):
    return ((polytropic_exponent - 1.0) /
            (polytropic_constant * polytropic_exponent) *
            (specific_enthalpy))**(1.0 / (polytropic_exponent - 1.0))


def polytropic_rel_specific_enthalpy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent):
    return 1.0 + polytropic_constant * polytropic_exponent / (
        polytropic_exponent - 1.0) * rest_mass_density**(
            polytropic_exponent - 1.0)


def polytropic_newt_specific_enthalpy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent):
    return polytropic_constant * polytropic_exponent / (
        polytropic_exponent - 1.0) * rest_mass_density**(
            polytropic_exponent - 1.0)


def polytropic_specific_internal_energy_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent):
    return polytropic_constant * rest_mass_density**(
        polytropic_exponent - 1.0) / (polytropic_exponent - 1.0)


def polytropic_chi_from_density(rest_mass_density, polytropic_constant,
                                polytropic_exponent):
    return polytropic_constant * polytropic_exponent * rest_mass_density**(
        polytropic_exponent - 1.0)


def polytropic_kappa_times_p_over_rho_squared_from_density(
        rest_mass_density, polytropic_constant, polytropic_exponent):
    return 0.0


# End functions for testing PolytropicFluid.cpp
