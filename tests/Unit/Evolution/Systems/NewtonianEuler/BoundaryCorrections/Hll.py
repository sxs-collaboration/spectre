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
