# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import Evolution.Systems.RelativisticEuler.Valencia.TestFunctions as valencia


def dg_package_data_tilde_d(
    tilde_d, tilde_tau, tilde_s, flux_tilde_d, flux_tilde_tau, flux_tilde_s,
    lapse, shift, spatial_metric, rest_mass_density, specific_internal_energy,
    specific_enthalpy, spatial_velocity, normal_covector, normal_vector,
    mesh_velocity, normal_dot_mesh_velocity, use_polytropic_fluid):
    return tilde_d


def dg_package_data_tilde_tau(
    tilde_d, tilde_tau, tilde_s, flux_tilde_d, flux_tilde_tau, flux_tilde_s,
    lapse, shift, spatial_metric, rest_mass_density, specific_internal_energy,
    specific_enthalpy, spatial_velocity, normal_covector, normal_vector,
    mesh_velocity, normal_dot_mesh_velocity, use_polytropic_fluid):
    return tilde_tau


def dg_package_data_tilde_s(
    tilde_d, tilde_tau, tilde_s, flux_tilde_d, flux_tilde_tau, flux_tilde_s,
    lapse, shift, spatial_metric, rest_mass_density, specific_internal_energy,
    specific_enthalpy, spatial_velocity, normal_covector, normal_vector,
    mesh_velocity, normal_dot_mesh_velocity, use_polytropic_fluid):
    return tilde_s


def dg_package_data_normal_dot_flux_tilde_d(
    tilde_d, tilde_tau, tilde_s, flux_tilde_d, flux_tilde_tau, flux_tilde_s,
    lapse, shift, spatial_metric, rest_mass_density, specific_internal_energy,
    specific_enthalpy, spatial_velocity, normal_covector, normal_vector,
    mesh_velocity, normal_dot_mesh_velocity, use_polytropic_fluid):
    return np.einsum("i,i", normal_covector, flux_tilde_d)


def dg_package_data_normal_dot_flux_tilde_tau(
    tilde_d, tilde_tau, tilde_s, flux_tilde_d, flux_tilde_tau, flux_tilde_s,
    lapse, shift, spatial_metric, rest_mass_density, specific_internal_energy,
    specific_enthalpy, spatial_velocity, normal_covector, normal_vector,
    mesh_velocity, normal_dot_mesh_velocity, use_polytropic_fluid):
    return np.einsum("i,i", normal_covector, flux_tilde_tau)


def dg_package_data_normal_dot_flux_tilde_s(
    tilde_d, tilde_tau, tilde_s, flux_tilde_d, flux_tilde_tau, flux_tilde_s,
    lapse, shift, spatial_metric, rest_mass_density, specific_internal_energy,
    specific_enthalpy, spatial_velocity, normal_covector, normal_vector,
    mesh_velocity, normal_dot_mesh_velocity, use_polytropic_fluid):
    return np.einsum("i,ij->j", normal_covector, flux_tilde_s)


def dg_package_data_abs_char_speed(
    tilde_d, tilde_tau, tilde_s, flux_tilde_d, flux_tilde_tau, flux_tilde_s,
    lapse, shift, spatial_metric, rest_mass_density, specific_internal_energy,
    specific_enthalpy, spatial_velocity, normal_covector, normal_vector,
    mesh_velocity, normal_dot_mesh_velocity, use_polytropic_fluid):

    spatial_velocity_squared = np.einsum("ij,i,j", spatial_metric,
                                         spatial_velocity, spatial_velocity)

    # Note that the relativistic sound speed squared has a 1/enthalpy
    if use_polytropic_fluid:
        polytropic_constant = 1.0e-3
        polytropic_exponent = 2.0
        sound_speed_squared = polytropic_constant * polytropic_exponent * pow(
            rest_mass_density, polytropic_exponent - 1.0) / specific_enthalpy
    else:
        adiabatic_index = 1.3
        chi = specific_internal_energy * (adiabatic_index - 1.0)
        kappa_times_p_over_rho_squared = ((adiabatic_index - 1.0)**2 *
                                          specific_internal_energy)
        sound_speed_squared = (
            chi + kappa_times_p_over_rho_squared) / specific_enthalpy

    char_speeds = valencia.characteristic_speeds(lapse, shift,
                                                 spatial_velocity,
                                                 spatial_velocity_squared,
                                                 sound_speed_squared,
                                                 normal_covector)

    if normal_dot_mesh_velocity is None:
        return np.max(np.abs(char_speeds))
    else:
        return np.max(np.abs(char_speeds - normal_dot_mesh_velocity))


def dg_boundary_terms_tilde_d(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s,
    interior_normal_dot_flux_tilde_d, interior_normal_dot_flux_tilde_tau,
    interior_normal_dot_flux_tilde_s, interior_abs_char_speed,
    exterior_tilde_d, exterior_tilde_tau, exterior_tilde_s,
    exterior_normal_dot_flux_tilde_d, exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s, exterior_abs_char_speed,
    use_strong_form):
    if use_strong_form:
        return (-0.5 * (interior_normal_dot_flux_tilde_d +
                        exterior_normal_dot_flux_tilde_d) - 0.5 *
                np.maximum(interior_abs_char_speed, exterior_abs_char_speed) *
                (exterior_tilde_d - interior_tilde_d))
    else:
        return (0.5 * (interior_normal_dot_flux_tilde_d -
                       exterior_normal_dot_flux_tilde_d) - 0.5 *
                np.maximum(interior_abs_char_speed, exterior_abs_char_speed) *
                (exterior_tilde_d - interior_tilde_d))


def dg_boundary_terms_tilde_tau(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s,
    interior_normal_dot_flux_tilde_d, interior_normal_dot_flux_tilde_tau,
    interior_normal_dot_flux_tilde_s, interior_abs_char_speed,
    exterior_tilde_d, exterior_tilde_tau, exterior_tilde_s,
    exterior_normal_dot_flux_tilde_d, exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s, exterior_abs_char_speed,
    use_strong_form):
    if use_strong_form:
        return (-0.5 * (interior_normal_dot_flux_tilde_tau +
                        exterior_normal_dot_flux_tilde_tau) - 0.5 *
                np.maximum(interior_abs_char_speed, exterior_abs_char_speed) *
                (exterior_tilde_tau - interior_tilde_tau))
    else:
        return (0.5 * (interior_normal_dot_flux_tilde_tau -
                       exterior_normal_dot_flux_tilde_tau) - 0.5 *
                np.maximum(interior_abs_char_speed, exterior_abs_char_speed) *
                (exterior_tilde_tau - interior_tilde_tau))


def dg_boundary_terms_tilde_s(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s,
    interior_normal_dot_flux_tilde_d, interior_normal_dot_flux_tilde_tau,
    interior_normal_dot_flux_tilde_s, interior_abs_char_speed,
    exterior_tilde_d, exterior_tilde_tau, exterior_tilde_s,
    exterior_normal_dot_flux_tilde_d, exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s, exterior_abs_char_speed,
    use_strong_form):
    if use_strong_form:
        return (-0.5 * (interior_normal_dot_flux_tilde_s +
                        exterior_normal_dot_flux_tilde_s) - 0.5 *
                np.maximum(interior_abs_char_speed, exterior_abs_char_speed) *
                (exterior_tilde_s - interior_tilde_s))
    else:
        return (0.5 * (interior_normal_dot_flux_tilde_s -
                       exterior_normal_dot_flux_tilde_s) - 0.5 *
                np.maximum(interior_abs_char_speed, exterior_abs_char_speed) *
                (exterior_tilde_s - interior_tilde_s))
