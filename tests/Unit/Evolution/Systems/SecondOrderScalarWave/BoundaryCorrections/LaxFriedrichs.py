# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    psi,
    pi,
    phi,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    tau,
):
    return (
        pi,
        np.asarray(np.einsum("i,i", normal_covector, phi)),
    )


def dg_boundary_terms(
    pi_int,
    normal_dot_phi_int,
    pi_ext,
    normal_dot_phi_ext,
    use_strong_form,
    tau,
):
    return (
        np.asarray(0.0 * pi_int),
        np.asarray(
            -0.5 * (normal_dot_phi_int + normal_dot_phi_ext)
            - tau * 0.5 * (pi_ext - pi_int)
        ),
    )


def dg_auxiliary_package_data(
    psi,
    pi,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    tau,
):
    return (np.asarray(psi * normal_covector),)


def dg_auxiliary_boundary_terms(
    psi_times_normal_int,
    psi_times_normal_ext,
    use_strong_form,
    tau,
):
    return (np.asarray(0.5 * (psi_times_normal_int + psi_times_normal_ext)),)
