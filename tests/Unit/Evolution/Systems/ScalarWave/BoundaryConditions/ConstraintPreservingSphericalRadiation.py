# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(face_mesh_velocity, outward_directed_normal_covector, pi, phi, psi,
          coords, interior_gamma2, dt_pi, dt_phi, dt_psi, d_pi, d_psi, d_phi):
    if not face_mesh_velocity is None and -np.dot(
            face_mesh_velocity, outward_directed_normal_covector) < 0.0:
        return ("Incoming characteristic speeds for constraint preserving "
                "spherical radiation boundary.*")
    return None


def _dt_psi(face_mesh_velocity, outward_directed_normal_covector, phi, d_psi):
    if face_mesh_velocity is None:
        return 0.0

    return -np.dot(outward_directed_normal_covector,
                   face_mesh_velocity) * np.dot(
                       outward_directed_normal_covector, d_psi - phi)


def dt_pi_Sommerfeld(face_mesh_velocity, outward_directed_normal_covector, pi,
                     phi, psi, coords, interior_gamma2, dt_pi, dt_phi, dt_psi,
                     d_pi, d_psi, d_phi):
    return -dt_pi + np.dot(
        outward_directed_normal_covector, dt_phi) + interior_gamma2 * _dt_psi(
            face_mesh_velocity, outward_directed_normal_covector, phi, d_psi)


def dt_pi_FirstOrderBaylissTurkel(face_mesh_velocity,
                                  outward_directed_normal_covector, pi, phi,
                                  psi, coords, interior_gamma2, dt_pi, dt_phi,
                                  dt_psi, d_pi, d_psi, d_phi):
    radius = np.sqrt(np.dot(coords, coords))
    return -dt_pi + np.dot(
        outward_directed_normal_covector,
        dt_phi) + dt_psi / radius + interior_gamma2 * _dt_psi(
            face_mesh_velocity, outward_directed_normal_covector, phi, d_psi)


def dt_pi_SecondOrderBaylissTurkel(face_mesh_velocity,
                                   outward_directed_normal_covector, pi, phi,
                                   psi, coords, interior_gamma2, dt_pi, dt_phi,
                                   dt_psi, d_pi, d_psi, d_phi):
    radius = np.sqrt(np.dot(coords, coords))
    return -dt_pi + (
        np.dot(outward_directed_normal_covector, dt_phi - d_pi) + np.einsum(
            "i,j,ij->", outward_directed_normal_covector,
            outward_directed_normal_covector, d_phi) - 4.0 / radius * pi +
        4.0 / radius * np.dot(outward_directed_normal_covector, phi) +
        2.0 * psi / radius**2) + interior_gamma2 * _dt_psi(
            face_mesh_velocity, outward_directed_normal_covector, phi, d_psi)


def dt_phi(face_mesh_velocity, outward_directed_normal_covector, pi, phi, psi,
           coords, interior_gamma2, dt_pi, dt_phi, dt_psi, d_pi, d_psi, d_phi):
    if face_mesh_velocity is None:
        return 0.0 * dt_phi

    return -np.dot(outward_directed_normal_covector,
                   face_mesh_velocity) * 0.5 * np.einsum(
                       "i, ij->j", outward_directed_normal_covector,
                       d_phi - np.transpose(d_phi))


def dt_psi(face_mesh_velocity, outward_directed_normal_covector, pi, phi, psi,
           coords, interior_gamma2, dt_pi, dt_phi, dt_psi, d_pi, d_psi, d_phi):
    assert interior_gamma2 >= 0.0  # make sure random gamma_2 is positive
    return _dt_psi(face_mesh_velocity, outward_directed_normal_covector, phi,
                   d_psi)
