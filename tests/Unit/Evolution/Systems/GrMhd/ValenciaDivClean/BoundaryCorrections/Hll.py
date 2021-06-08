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


def dg_package_data_largest_outgoing_char_speed(
    tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, flux_tilde_d,
    flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
    normal_covector, normal_vector, mesh_velocity, normal_dot_mesh_velocity):
    if normal_dot_mesh_velocity is None:
        return lapse - np.dot(shift, normal_covector)
    else:
        return lapse - np.dot(shift,
                              normal_covector) - normal_dot_mesh_velocity


def dg_package_data_largest_ingoing_char_speed(
    tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, flux_tilde_d,
    flux_tilde_tau, flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
    normal_covector, normal_vector, mesh_velocity, normal_dot_mesh_velocity):
    if normal_dot_mesh_velocity is None:
        return -lapse - np.dot(shift, normal_covector)
    else:
        return -lapse - np.dot(shift,
                               normal_covector) - normal_dot_mesh_velocity


def dg_boundary_terms_tilde_d(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_largest_outgoing_char_speed, interior_largest_ingoing_char_speed,
    exterior_tilde_d, exterior_tilde_tau, exterior_tilde_s, exterior_tilde_b,
    exterior_tilde_phi, exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_tau, exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b, exterior_normal_dot_flux_tilde_phi,
    exterior_largest_outgoing_char_speed, exterior_largest_ingoing_char_speed,
    use_strong_form):

    lambda_max = np.maximum(
        0.,
        np.maximum(interior_largest_outgoing_char_speed,
                   -exterior_largest_ingoing_char_speed))
    lambda_min = np.minimum(
        0.,
        np.minimum(interior_largest_ingoing_char_speed,
                   -exterior_largest_outgoing_char_speed))

    if use_strong_form:
        return (lambda_min *
                (interior_normal_dot_flux_tilde_d +
                 exterior_normal_dot_flux_tilde_d) + lambda_max * lambda_min *
                (exterior_tilde_d - interior_tilde_d)) / (lambda_max -
                                                          lambda_min)
    else:
        return (
            (lambda_max * interior_normal_dot_flux_tilde_d + lambda_min *
             exterior_normal_dot_flux_tilde_d) + lambda_max * lambda_min *
            (exterior_tilde_d - interior_tilde_d)) / (lambda_max - lambda_min)


def dg_boundary_terms_tilde_tau(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_largest_outgoing_char_speed, interior_largest_ingoing_char_speed,
    exterior_tilde_d, exterior_tilde_tau, exterior_tilde_s, exterior_tilde_b,
    exterior_tilde_phi, exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_tau, exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b, exterior_normal_dot_flux_tilde_phi,
    exterior_largest_outgoing_char_speed, exterior_largest_ingoing_char_speed,
    use_strong_form):

    lambda_max = np.maximum(
        0.,
        np.maximum(interior_largest_outgoing_char_speed,
                   -exterior_largest_ingoing_char_speed))
    lambda_min = np.minimum(
        0.,
        np.minimum(interior_largest_ingoing_char_speed,
                   -exterior_largest_outgoing_char_speed))

    if use_strong_form:
        return (lambda_min * (interior_normal_dot_flux_tilde_tau +
                              exterior_normal_dot_flux_tilde_tau) +
                lambda_max * lambda_min *
                (exterior_tilde_tau - interior_tilde_tau)) / (lambda_max -
                                                              lambda_min)
    else:
        return ((lambda_max * interior_normal_dot_flux_tilde_tau +
                 lambda_min * exterior_normal_dot_flux_tilde_tau) +
                lambda_max * lambda_min *
                (exterior_tilde_tau - interior_tilde_tau)) / (lambda_max -
                                                              lambda_min)


def dg_boundary_terms_tilde_s(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_largest_outgoing_char_speed, interior_largest_ingoing_char_speed,
    exterior_tilde_d, exterior_tilde_tau, exterior_tilde_s, exterior_tilde_b,
    exterior_tilde_phi, exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_tau, exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b, exterior_normal_dot_flux_tilde_phi,
    exterior_largest_outgoing_char_speed, exterior_largest_ingoing_char_speed,
    use_strong_form):

    lambda_max = np.maximum(
        0.,
        np.maximum(interior_largest_outgoing_char_speed,
                   -exterior_largest_ingoing_char_speed))
    lambda_min = np.minimum(
        0.,
        np.minimum(interior_largest_ingoing_char_speed,
                   -exterior_largest_outgoing_char_speed))

    if use_strong_form:
        return (lambda_min *
                (interior_normal_dot_flux_tilde_s +
                 exterior_normal_dot_flux_tilde_s) + lambda_max * lambda_min *
                (exterior_tilde_s - interior_tilde_s)) / (lambda_max -
                                                          lambda_min)
    else:
        return (
            (lambda_max * interior_normal_dot_flux_tilde_s + lambda_min *
             exterior_normal_dot_flux_tilde_s) + lambda_max * lambda_min *
            (exterior_tilde_s - interior_tilde_s)) / (lambda_max - lambda_min)


def dg_boundary_terms_tilde_b(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_largest_outgoing_char_speed, interior_largest_ingoing_char_speed,
    exterior_tilde_d, exterior_tilde_tau, exterior_tilde_s, exterior_tilde_b,
    exterior_tilde_phi, exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_tau, exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b, exterior_normal_dot_flux_tilde_phi,
    exterior_largest_outgoing_char_speed, exterior_largest_ingoing_char_speed,
    use_strong_form):

    lambda_max = np.maximum(
        0.,
        np.maximum(interior_largest_outgoing_char_speed,
                   -exterior_largest_ingoing_char_speed))
    lambda_min = np.minimum(
        0.,
        np.minimum(interior_largest_ingoing_char_speed,
                   -exterior_largest_outgoing_char_speed))

    if use_strong_form:
        return (lambda_min *
                (interior_normal_dot_flux_tilde_b +
                 exterior_normal_dot_flux_tilde_b) + lambda_max * lambda_min *
                (exterior_tilde_b - interior_tilde_b)) / (lambda_max -
                                                          lambda_min)
    else:
        return (
            (lambda_max * interior_normal_dot_flux_tilde_b + lambda_min *
             exterior_normal_dot_flux_tilde_b) + lambda_max * lambda_min *
            (exterior_tilde_b - interior_tilde_b)) / (lambda_max - lambda_min)


def dg_boundary_terms_tilde_phi(
    interior_tilde_d, interior_tilde_tau, interior_tilde_s, interior_tilde_b,
    interior_tilde_phi, interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_tau, interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b, interior_normal_dot_flux_tilde_phi,
    interior_largest_outgoing_char_speed, interior_largest_ingoing_char_speed,
    exterior_tilde_d, exterior_tilde_tau, exterior_tilde_s, exterior_tilde_b,
    exterior_tilde_phi, exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_tau, exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b, exterior_normal_dot_flux_tilde_phi,
    exterior_largest_outgoing_char_speed, exterior_largest_ingoing_char_speed,
    use_strong_form):

    lambda_max = np.maximum(
        0.,
        np.maximum(interior_largest_outgoing_char_speed,
                   -exterior_largest_ingoing_char_speed))
    lambda_min = np.minimum(
        0.,
        np.minimum(interior_largest_ingoing_char_speed,
                   -exterior_largest_outgoing_char_speed))

    if use_strong_form:
        return (lambda_min * (interior_normal_dot_flux_tilde_phi +
                              exterior_normal_dot_flux_tilde_phi) +
                lambda_max * lambda_min *
                (exterior_tilde_phi - interior_tilde_phi)) / (lambda_max -
                                                              lambda_min)
    else:
        return ((lambda_max * interior_normal_dot_flux_tilde_phi +
                 lambda_min * exterior_normal_dot_flux_tilde_phi) +
                lambda_max * lambda_min *
                (exterior_tilde_phi - interior_tilde_phi)) / (lambda_max -
                                                              lambda_min)
