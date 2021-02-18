# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data_mass_density(
    mass_density, momentum_density, energy_density, flux_mass_density,
    flux_momentum_density, flux_energy_density, velocity,
    specific_internal_energy, normal_covector, mesh_velocity,
    normal_dot_mesh_velocity, use_polytropic_fluid):
    return mass_density


def dg_package_data_momentum_density(
    mass_density, momentum_density, energy_density, flux_mass_density,
    flux_momentum_density, flux_energy_density, velocity,
    specific_internal_energy, normal_covector, mesh_velocity,
    normal_dot_mesh_velocity, use_polytropic_fluid):
    return momentum_density


def dg_package_data_energy_density(
    mass_density, momentum_density, energy_density, flux_mass_density,
    flux_momentum_density, flux_energy_density, velocity,
    specific_internal_energy, normal_covector, mesh_velocity,
    normal_dot_mesh_velocity, use_polytropic_fluid):
    return energy_density


def dg_package_data_normal_dot_flux_mass_density(
    mass_density, momentum_density, energy_density, flux_mass_density,
    flux_momentum_density, flux_energy_density, velocity,
    specific_internal_energy, normal_covector, mesh_velocity,
    normal_dot_mesh_velocity, use_polytropic_fluid):
    return np.einsum("i,i", normal_covector, flux_mass_density)


def dg_package_data_normal_dot_flux_momentum_density(
    mass_density, momentum_density, energy_density, flux_mass_density,
    flux_momentum_density, flux_energy_density, velocity,
    specific_internal_energy, normal_covector, mesh_velocity,
    normal_dot_mesh_velocity, use_polytropic_fluid):
    return np.einsum("i,ij->j", normal_covector, flux_momentum_density)


def dg_package_data_normal_dot_flux_energy_density(
    mass_density, momentum_density, energy_density, flux_mass_density,
    flux_momentum_density, flux_energy_density, velocity,
    specific_internal_energy, normal_covector, mesh_velocity,
    normal_dot_mesh_velocity, use_polytropic_fluid):
    return np.einsum("i,i", normal_covector, flux_energy_density)


def dg_package_data_largest_outgoing_char_speed(
    mass_density, momentum_density, energy_density, flux_mass_density,
    flux_momentum_density, flux_energy_density, velocity,
    specific_internal_energy, normal_covector, mesh_velocity,
    normal_dot_mesh_velocity, use_polytropic_fluid):
    velocity_dot_normal = np.einsum("i,i", normal_covector, velocity)
    if use_polytropic_fluid:
        polytropic_constant = 1.0e-3
        polytropic_exponent = 2.0
        sound_speed = np.sqrt(polytropic_constant * polytropic_exponent *
                              pow(mass_density, polytropic_exponent - 1.0))
    else:
        adiabatic_index = 1.3
        chi = specific_internal_energy * (adiabatic_index - 1.0)
        kappa_times_p_over_rho_squared = ((adiabatic_index - 1.0)**2 *
                                          specific_internal_energy)
        sound_speed = np.sqrt(chi + kappa_times_p_over_rho_squared)

    if normal_dot_mesh_velocity is None:
        return velocity_dot_normal + sound_speed
    else:
        return velocity_dot_normal + sound_speed - normal_dot_mesh_velocity


def dg_package_data_largest_ingoing_char_speed(
    mass_density, momentum_density, energy_density, flux_mass_density,
    flux_momentum_density, flux_energy_density, velocity,
    specific_internal_energy, normal_covector, mesh_velocity,
    normal_dot_mesh_velocity, use_polytropic_fluid):
    velocity_dot_normal = np.einsum("i,i", normal_covector, velocity)
    if use_polytropic_fluid:
        polytropic_constant = 1.0e-3
        polytropic_exponent = 2.0
        sound_speed = np.sqrt(polytropic_constant * polytropic_exponent *
                              pow(mass_density, polytropic_exponent - 1.0))
    else:
        adiabatic_index = 1.3
        chi = specific_internal_energy * (adiabatic_index - 1.0)
        kappa_times_p_over_rho_squared = ((adiabatic_index - 1.0)**2 *
                                          specific_internal_energy)
        sound_speed = np.sqrt(chi + kappa_times_p_over_rho_squared)

    if normal_dot_mesh_velocity is None:
        return velocity_dot_normal - sound_speed
    else:
        return velocity_dot_normal - sound_speed - normal_dot_mesh_velocity
