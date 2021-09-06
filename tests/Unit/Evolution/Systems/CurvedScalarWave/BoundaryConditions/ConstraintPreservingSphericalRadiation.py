# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Evolution.Systems.CurvedScalarWave.Characteristics import (
    char_speed_vpsi, char_speed_vzero, char_speed_vplus, char_speed_vminus)


def error(face_mesh_velocity, normal_covector, normal_vector, phi, psi,
          inertial_coords, gamma1, gamma2, lapse, shift, dt_pi, dt_phi, dt_psi,
          d_psi, d_phi):
    return None


def dt_psi_constraint_preserving_spherical_radiation(
    face_mesh_velocity, normal_covector, normal_vector, phi, psi,
    inertial_coords, gamma1, gamma2, lapse, shift, dt_pi, dt_phi, dt_psi,
    d_psi, d_phi):
    char_speed_psi = char_speed_vpsi(gamma1, lapse, shift, normal_covector)

    if face_mesh_velocity is not None:
        char_speed_psi -= np.dot(normal_covector, face_mesh_velocity)

    return np.dot(normal_vector, d_psi - phi) * min(0., char_speed_psi)


def dt_phi_constraint_preserving_spherical_radiation(
    face_mesh_velocity, normal_covector, normal_vector, phi, psi,
    inertial_coords, gamma1, gamma2, lapse, shift, dt_pi, dt_phi, dt_psi,
    d_psi, d_phi):
    char_speed_zero = char_speed_vzero(gamma1, lapse, shift, normal_covector)
    if face_mesh_velocity is not None:
        char_speed_zero -= np.dot(normal_covector, face_mesh_velocity)
    return 0.5 * np.einsum("ij,j", d_phi.T - d_phi, normal_vector) * min(
        0, char_speed_zero)


def dt_pi_constraint_preserving_spherical_radiation(
    face_mesh_velocity, normal_covector, normal_vector, phi, psi,
    inertial_coords, gamma1, gamma2, lapse, shift, dt_pi, dt_phi, dt_psi,
    d_psi, d_phi):
    dt_psi_correction = dt_psi_constraint_preserving_spherical_radiation(
        face_mesh_velocity, normal_covector, normal_vector, phi, psi,
        inertial_coords, gamma1, gamma2, lapse, shift, dt_pi, dt_phi, dt_psi,
        d_psi, d_phi)
    inv_radius = 1. / np.linalg.norm(inertial_coords)
    bc_dt_pi = (2. * inv_radius**2 * psi + 4. * inv_radius * dt_psi +
                4. * inv_radius * np.dot(normal_vector, phi) +
                2. * np.dot(normal_vector, dt_phi) + np.dot(shift, dt_phi) +
                np.einsum("i,j,ij", normal_vector, normal_vector, d_phi))
    bc_dt_pi /= lapse
    return bc_dt_pi - dt_pi + gamma2 * dt_psi_correction
