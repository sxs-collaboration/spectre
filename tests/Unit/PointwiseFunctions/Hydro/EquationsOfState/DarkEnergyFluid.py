# Distributed under the MIT License.
# See LICENSE.txt for details.


def dark_energy_fluid_pressure_from_density_and_energy(
    rest_mass_density, specific_internal_energy, parameter_w
):
    return parameter_w * rest_mass_density * (1.0 + specific_internal_energy)


def dark_energy_fluid_rel_pressure_from_density_and_enthalpy(
    rest_mass_density, specific_enthalpy, parameter_w
):
    return (
        parameter_w
        * rest_mass_density
        * specific_enthalpy
        / (parameter_w + 1.0)
    )


def dark_energy_fluid_rel_specific_enthalpy_from_density_and_energy(
    rest_mass_density, specific_internal_energy, parameter_w
):
    return (parameter_w + 1.0) * (1.0 + specific_internal_energy)


def dark_energy_fluid_temperature_from_density_and_energy(
    rest_mass_density, specific_internal_energy, parameter_w
):
    return parameter_w * specific_internal_energy


def dark_energy_fluid_specific_internal_energy_from_density_and_pressure(
    rest_mass_density, pressure, parameter_w
):
    return pressure / (parameter_w * rest_mass_density) - 1.0


def dark_energy_fluid_chi_from_density_and_energy(
    rest_mass_density, specific_internal_energy, parameter_w
):
    return parameter_w * (1.0 + specific_internal_energy)


def dark_energy_fluid_kappa_times_p_over_rho_squared_from_density_and_energy(
    rest_mass_density, specific_internal_energy, parameter_w
):
    return parameter_w**2 * (1.0 + specific_internal_energy)
