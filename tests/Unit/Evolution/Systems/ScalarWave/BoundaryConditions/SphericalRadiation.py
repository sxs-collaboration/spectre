# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(face_mesh_velocity, outward_directed_normal_covector, phi, psi,
          coords, interior_gamma2):
    if not face_mesh_velocity is None and -np.dot(
            face_mesh_velocity, outward_directed_normal_covector) < 0.0:
        return (
            "Incoming characteristic speeds for spherical radiation boundary.*"
        )
    return None


def pi_Sommerfeld(face_mesh_velocity, outward_directed_normal_covector, phi,
                  psi, coords, interior_gamma2):
    return np.dot(outward_directed_normal_covector, phi)


def pi_BaylissTurkel(face_mesh_velocity, outward_directed_normal_covector, phi,
                     psi, coords, interior_gamma2):
    radius = np.sqrt(np.dot(coords, coords))
    return np.dot(outward_directed_normal_covector, phi) + psi / radius


def phi(face_mesh_velocity, outward_directed_normal_covector, phi, psi, coords,
        interior_gamma2):
    return phi


def psi(face_mesh_velocity, outward_directed_normal_covector, phi, psi, coords,
        interior_gamma2):
    return psi


def constraint_gamma2(face_mesh_velocity, outward_directed_normal_covector,
                      phi, psi, coords, interior_gamma2):
    assert interior_gamma2 >= 0.0  # make sure random gamma_2 is positive
    return interior_gamma2
