# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    pi,
    phi,
    psi,
    coords,
    interior_gamma2,
    d_psi,
    d_pi,
    d_phi,
):
    if (
        not face_mesh_velocity is None
        and -np.dot(face_mesh_velocity, outward_directed_normal_covector) < 0.0
    ):
        return (
            "Incoming characteristic speeds for constraint preserving "
            "spherical radiation boundary.*"
        )
    return None


def _dt_psi(face_mesh_velocity, outward_directed_normal_covector, phi, d_psi):
    if face_mesh_velocity is None:
        return 0.0

    return -np.dot(
        outward_directed_normal_covector, face_mesh_velocity
    ) * np.dot(outward_directed_normal_covector, d_psi - phi)


def dt_pi_Sommerfeld(
    face_mesh_velocity,
    outward_directed_normal_covector,
    psi,
    pi,
    phi,
    coords,
    interior_gamma2,
    d_psi,
    d_pi,
    d_phi,
):
    return (
        np.einsum("ii", d_phi)
        + np.einsum(
            "i,i->",
            outward_directed_normal_covector,
            interior_gamma2 * (d_psi - phi) - d_pi,
        )
        + interior_gamma2
        * _dt_psi(
            face_mesh_velocity, outward_directed_normal_covector, phi, d_psi
        )
    )


def dt_pi_FirstOrderBaylissTurkel(
    face_mesh_velocity,
    outward_directed_normal_covector,
    psi,
    pi,
    phi,
    coords,
    interior_gamma2,
    d_psi,
    d_pi,
    d_phi,
):
    radius = np.sqrt(np.dot(coords, coords))
    return (
        np.einsum("ii", d_phi)
        + np.einsum(
            "i,i->",
            outward_directed_normal_covector,
            interior_gamma2 * (d_psi - phi) - d_pi,
        )
        - pi / radius
        + interior_gamma2
        * _dt_psi(
            face_mesh_velocity, outward_directed_normal_covector, phi, d_psi
        )
    )


def dt_pi_SecondOrderBaylissTurkel(
    face_mesh_velocity,
    outward_directed_normal_covector,
    psi,
    pi,
    phi,
    coords,
    interior_gamma2,
    d_psi,
    d_pi,
    d_phi,
):
    radius = np.sqrt(np.dot(coords, coords))
    return (
        np.einsum("ii", d_phi)
        + (
            np.dot(
                outward_directed_normal_covector,
                -2.0 * d_pi + interior_gamma2 * (d_psi - phi),
            )
            + np.einsum(
                "i,j,ij->",
                outward_directed_normal_covector,
                outward_directed_normal_covector,
                d_phi,
            )
            - 4.0 / radius * pi
            + 4.0 / radius * np.dot(outward_directed_normal_covector, phi)
            + 2.0 * psi / radius**2
        )
        + interior_gamma2
        * _dt_psi(
            face_mesh_velocity, outward_directed_normal_covector, phi, d_psi
        )
    )


def dt_phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    psi,
    pi,
    phi,
    coords,
    interior_gamma2,
    d_psi,
    d_pi,
    d_phi,
):
    if face_mesh_velocity is None:
        return 0.0 * d_psi

    return -np.dot(
        outward_directed_normal_covector, face_mesh_velocity
    ) * np.einsum(
        "i, ij->j",
        outward_directed_normal_covector,
        d_phi - np.transpose(d_phi),
    )


def dt_psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    psi,
    pi,
    phi,
    coords,
    interior_gamma2,
    d_psi,
    d_pi,
    d_phi,
):
    assert interior_gamma2 >= 0.0  # make sure random gamma_2 is positive
    return _dt_psi(
        face_mesh_velocity, outward_directed_normal_covector, phi, d_psi
    )
