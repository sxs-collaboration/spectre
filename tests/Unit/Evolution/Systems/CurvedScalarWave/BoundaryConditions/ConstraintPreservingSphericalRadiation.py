# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Evolution.Systems.CurvedScalarWave.Characteristics import (
    char_speed_vminus,
    char_speed_vplus,
    char_speed_vpsi,
    char_speed_vzero,
)


def error(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    phi,
    inertial_coords,
    gamma1,
    gamma2,
    lapse,
    shift,
    dt_psi,
    dt_pi,
    dt_phi,
    d_psi,
    d_pi,
    d_phi,
):
    return None


def subtract_mesh_velocity(
    face_mesh_velocity, dt_psi, dt_pi, dt_phi, d_psi, d_pi, d_phi
):
    if not (face_mesh_velocity is None):
        dt_psi -= np.einsum("i,i->", face_mesh_velocity, d_psi)
        dt_pi -= np.einsum("i,i->", face_mesh_velocity, d_pi)
        dt_phi -= np.einsum("i,ij->j", face_mesh_velocity, d_phi)


def dt_psi_constraint_preserving_spherical_radiation(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    phi,
    inertial_coords,
    gamma1,
    gamma2,
    lapse,
    shift,
    dt_psi,
    dt_pi,
    dt_phi,
    d_psi,
    d_pi,
    d_phi,
):
    subtract_mesh_velocity(
        face_mesh_velocity, dt_psi, dt_pi, dt_phi, d_psi, d_pi, d_phi
    )
    char_speed_psi = char_speed_vpsi(gamma1, lapse, shift, normal_covector)

    if face_mesh_velocity is not None:
        char_speed_psi -= np.dot(normal_covector, face_mesh_velocity)

    return np.dot(normal_vector, d_psi - phi) * min(0.0, char_speed_psi)


def dt_phi_constraint_preserving_spherical_radiation(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    phi,
    inertial_coords,
    gamma1,
    gamma2,
    lapse,
    shift,
    dt_psi,
    dt_pi,
    dt_phi,
    d_psi,
    d_pi,
    d_phi,
):
    subtract_mesh_velocity(
        face_mesh_velocity, dt_psi, dt_pi, dt_phi, d_psi, d_pi, d_phi
    )
    char_speed_zero = char_speed_vzero(gamma1, lapse, shift, normal_covector)
    if face_mesh_velocity is not None:
        char_speed_zero -= np.dot(normal_covector, face_mesh_velocity)
    return np.einsum("ij,j", d_phi.T - d_phi, normal_vector) * min(
        0, char_speed_zero
    )


def dt_pi_constraint_preserving_spherical_radiation(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    phi,
    inertial_coords,
    gamma1,
    gamma2,
    lapse,
    shift,
    dt_psi,
    dt_pi,
    dt_phi,
    d_psi,
    d_pi,
    d_phi,
):
    dt_psi_correction = dt_psi_constraint_preserving_spherical_radiation(
        face_mesh_velocity,
        normal_covector,
        normal_vector,
        psi,
        phi,
        inertial_coords,
        gamma1,
        gamma2,
        lapse,
        shift,
        dt_psi,
        dt_pi,
        dt_phi,
        d_psi,
        d_pi,
        d_phi,
    )
    # dt's are already corrected in dt_psi
    inv_radius = 1.0 / np.linalg.norm(inertial_coords)
    bc_dt_pi = (
        2.0 * inv_radius**2 * psi
        + 4.0 * inv_radius * dt_psi
        + 4.0 * inv_radius * np.dot(normal_vector, phi)
        + 2.0 * np.dot(normal_vector, dt_phi)
        + np.dot(shift, dt_phi)
        + np.einsum("i,j,ij", normal_vector, normal_vector, d_phi)
    )
    bc_dt_pi /= lapse
    return bc_dt_pi - dt_pi + gamma2 * dt_psi_correction
