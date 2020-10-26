# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data_tilde_d(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                            flux_tilde_d, flux_tilde_tau, flux_tilde_s,
                            flux_tilde_b, flux_tilde_phi, lapse, shift,
                            normal_covector, normal_vector, mesh_velocity,
                            normal_dot_mesh_velocity):
    return tilde_d


def dg_package_data_tilde_tau(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                              flux_tilde_d, flux_tilde_tau, flux_tilde_s,
                              flux_tilde_b, flux_tilde_phi, lapse, shift,
                              normal_covector, normal_vector, mesh_velocity,
                              normal_dot_mesh_velocity):
    return tilde_tau


def dg_package_data_tilde_s(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                            flux_tilde_d, flux_tilde_tau, flux_tilde_s,
                            flux_tilde_b, flux_tilde_phi, lapse, shift,
                            normal_covector, normal_vector, mesh_velocity,
                            normal_dot_mesh_velocity):
    return tilde_s


def dg_package_data_tilde_b(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                            flux_tilde_d, flux_tilde_tau, flux_tilde_s,
                            flux_tilde_b, flux_tilde_phi, lapse, shift,
                            normal_covector, normal_vector, mesh_velocity,
                            normal_dot_mesh_velocity):
    return tilde_b


def dg_package_data_tilde_phi(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                              flux_tilde_d, flux_tilde_tau, flux_tilde_s,
                              flux_tilde_b, flux_tilde_phi, lapse, shift,
                              normal_covector, normal_vector, mesh_velocity,
                              normal_dot_mesh_velocity):
    return tilde_phi


def dg_package_data_normal_dot_flux_tilde_d(
    tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, flux_tilde_d,
    flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
    normal_covector, normal_vector, mesh_velocity, normal_dot_mesh_velocity):
    return np.dot(flux_tilde_d, normal_covector)


def dg_package_data_normal_dot_flux_tilde_tau(
    tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, flux_tilde_d,
    flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
    normal_covector, normal_vector, mesh_velocity, normal_dot_mesh_velocity):
    return np.dot(flux_tilde_tau, normal_covector)


def dg_package_data_normal_dot_flux_tilde_s(
    tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, flux_tilde_d,
    flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
    normal_covector, normal_vector, mesh_velocity, normal_dot_mesh_velocity):
    return np.einsum("ij,i->j", flux_tilde_s, normal_covector)


def dg_package_data_normal_dot_flux_tilde_b(
    tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, flux_tilde_d,
    flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
    normal_covector, normal_vector, mesh_velocity, normal_dot_mesh_velocity):
    return np.einsum("ij,i->j", flux_tilde_b, normal_covector)


def dg_package_data_normal_dot_flux_tilde_phi(
    tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, flux_tilde_d,
    flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
    normal_covector, normal_vector, mesh_velocity, normal_dot_mesh_velocity):
    return np.dot(flux_tilde_phi, normal_covector)


def dg_package_data_abs_char_speed(tilde_d, tilde_tau, tilde_s, tilde_b,
                                   tilde_phi, flux_tilde_d, flux_tilde_tau,
                                   flux_tilde_s, flux_tilde_b, flux_tilde_phi,
                                   lapse, shift, normal_covector,
                                   normal_vector, mesh_velocity,
                                   normal_dot_mesh_velocity):
    if normal_dot_mesh_velocity is None:
        return np.maximum(np.abs(lapse - np.dot(shift, normal_covector)),
                          np.abs(-lapse - np.dot(shift, normal_covector)))
    else:
        return np.maximum(
            np.abs(lapse - np.dot(shift, normal_covector) -
                   normal_dot_mesh_velocity),
            np.abs(-lapse - np.dot(shift, normal_covector) -
                   normal_dot_mesh_velocity))


def dg_boundary_terms_tilde_d(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_abs_char_speed, exterior_tilde_d, exterior_tilde_tau,
    exterior_tilde_s, exterior_tilde_b, exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d, exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s, exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi, exterior_abs_char_speed,
    use_strong_form):
    if use_strong_form:
        return -0.5 * (interior_normal_dot_flux_tilde_d +
                       exterior_normal_dot_flux_tilde_d) - 0.5 * np.maximum(
                           interior_abs_char_speed, exterior_abs_char_speed
                       ) * (exterior_tilde_d - interior_tilde_d)
    else:
        return 0.5 * (interior_normal_dot_flux_tilde_d -
                      exterior_normal_dot_flux_tilde_d) - 0.5 * np.maximum(
                          interior_abs_char_speed, exterior_abs_char_speed) * (
                              exterior_tilde_d - interior_tilde_d)


def dg_boundary_terms_tilde_tau(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_abs_char_speed, exterior_tilde_d, exterior_tilde_tau,
    exterior_tilde_s, exterior_tilde_b, exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d, exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s, exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi, exterior_abs_char_speed,
    use_strong_form):
    if use_strong_form:
        return -0.5 * (interior_normal_dot_flux_tilde_tau +
                       exterior_normal_dot_flux_tilde_tau) - 0.5 * np.maximum(
                           interior_abs_char_speed, exterior_abs_char_speed
                       ) * (exterior_tilde_tau - interior_tilde_tau)
    else:
        return 0.5 * (interior_normal_dot_flux_tilde_tau -
                      exterior_normal_dot_flux_tilde_tau) - 0.5 * np.maximum(
                          interior_abs_char_speed, exterior_abs_char_speed) * (
                              exterior_tilde_tau - interior_tilde_tau)


def dg_boundary_terms_tilde_s(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_abs_char_speed, exterior_tilde_d, exterior_tilde_tau,
    exterior_tilde_s, exterior_tilde_b, exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d, exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s, exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi, exterior_abs_char_speed,
    use_strong_form):
    if use_strong_form:
        return -0.5 * (interior_normal_dot_flux_tilde_s +
                       exterior_normal_dot_flux_tilde_s) - 0.5 * np.maximum(
                           interior_abs_char_speed, exterior_abs_char_speed
                       ) * (exterior_tilde_s - interior_tilde_s)
    else:
        return 0.5 * (interior_normal_dot_flux_tilde_s -
                      exterior_normal_dot_flux_tilde_s) - 0.5 * np.maximum(
                          interior_abs_char_speed, exterior_abs_char_speed) * (
                              exterior_tilde_s - interior_tilde_s)


def dg_boundary_terms_tilde_b(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_abs_char_speed, exterior_tilde_d, exterior_tilde_tau,
    exterior_tilde_s, exterior_tilde_b, exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d, exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s, exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi, exterior_abs_char_speed,
    use_strong_form):
    if use_strong_form:
        return -0.5 * (interior_normal_dot_flux_tilde_b +
                       exterior_normal_dot_flux_tilde_b) - 0.5 * np.maximum(
                           interior_abs_char_speed, exterior_abs_char_speed
                       ) * (exterior_tilde_b - interior_tilde_b)
    else:
        return 0.5 * (interior_normal_dot_flux_tilde_b -
                      exterior_normal_dot_flux_tilde_b) - 0.5 * np.maximum(
                          interior_abs_char_speed, exterior_abs_char_speed) * (
                              exterior_tilde_b - interior_tilde_b)


def dg_boundary_terms_tilde_phi(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_abs_char_speed, exterior_tilde_d, exterior_tilde_tau,
    exterior_tilde_s, exterior_tilde_b, exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d, exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s, exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi, exterior_abs_char_speed,
    use_strong_form):
    if use_strong_form:
        return -0.5 * (interior_normal_dot_flux_tilde_phi +
                       exterior_normal_dot_flux_tilde_phi) - 0.5 * np.maximum(
                           interior_abs_char_speed, exterior_abs_char_speed
                       ) * (exterior_tilde_phi - interior_tilde_phi)
    else:
        return 0.5 * (interior_normal_dot_flux_tilde_phi -
                      exterior_normal_dot_flux_tilde_phi) - 0.5 * np.maximum(
                          interior_abs_char_speed, exterior_abs_char_speed) * (
                              exterior_tilde_phi - interior_tilde_phi)
