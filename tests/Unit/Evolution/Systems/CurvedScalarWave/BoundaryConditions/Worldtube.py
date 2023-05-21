# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from Evolution.Systems.CurvedScalarWave import Characteristics


def error(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    return None


def dt_psi_worldtube(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    char_speed_psi = Characteristics.char_speed_vpsi(
        gamma1, lapse, shift, normal_covector
    )

    if face_mesh_velocity is not None:
        char_speed_psi -= np.dot(normal_covector, face_mesh_velocity)

    return np.dot(normal_vector, d_psi - phi) * min(0.0, char_speed_psi)


def dt_phi_worldtube(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    char_speed_zero = Characteristics.char_speed_vzero(
        gamma1, lapse, shift, normal_covector
    )
    if face_mesh_velocity is not None:
        char_speed_zero -= np.dot(normal_covector, face_mesh_velocity)
    return np.einsum("ij,j", d_phi.T - d_phi, normal_vector) * min(
        0, char_speed_zero
    )


def dt_pi_worldtube(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    return gamma2 * dt_psi_worldtube(
        face_mesh_velocity,
        normal_covector,
        normal_vector,
        psi,
        pi,
        phi,
        lapse,
        shift,
        inverse_spatial_metric,
        gamma1,
        gamma2,
        dt_psi,
        d_psi,
        d_phi,
        worldtube_vars,
        dim,
    )


def psi(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    return psi


def pi(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    # The worldtube data on the C++ side is a Variables filled with ones.
    # Since we do not want a Variables on the python side, we hard-code it here
    wt_psi = 1.0
    wt_pi = 1.0
    wt_phi = np.ones(3)

    vpsi_interior = Characteristics.char_field_vpsi(
        gamma2, psi, pi, phi, normal_covector, normal_vector
    )
    vzero_interior = Characteristics.char_field_vzero(
        gamma2, psi, pi, phi, normal_covector, normal_vector
    )
    vplus_interior = Characteristics.char_field_vplus(
        gamma2, psi, pi, phi, normal_covector, normal_vector
    )
    vminus_wt = Characteristics.char_field_vminus(
        gamma2, wt_psi, wt_pi, wt_phi, normal_covector, normal_vector
    )

    return Characteristics.evol_field_pi(
        gamma2,
        vpsi_interior,
        vzero_interior,
        vplus_interior,
        vminus_wt,
        normal_covector,
    )


def phi(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    # The worldtube data on the C++ side is a Variables filled with ones.
    # Since we do not want a Variables on the python side, we hard-code it here
    wt_psi = 1.0
    wt_pi = 1.0
    wt_phi = np.ones(3)

    vpsi_interior = Characteristics.char_field_vpsi(
        gamma2, psi, pi, phi, normal_covector, normal_vector
    )
    vzero_interior = Characteristics.char_field_vzero(
        gamma2, psi, pi, phi, normal_covector, normal_vector
    )
    vplus_interior = Characteristics.char_field_vplus(
        gamma2, psi, pi, phi, normal_covector, normal_vector
    )
    vminus_wt = Characteristics.char_field_vminus(
        gamma2, wt_psi, wt_pi, wt_phi, normal_covector, normal_vector
    )
    return Characteristics.evol_field_phi(
        gamma2,
        vpsi_interior,
        vzero_interior,
        vplus_interior,
        vminus_wt,
        normal_covector,
    )


def lapse(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    return lapse


def shift(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    return shift


def gamma1(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    return gamma1


def gamma2(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    return gamma2


def inverse_spatial_metric(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    psi,
    pi,
    phi,
    lapse,
    shift,
    inverse_spatial_metric,
    gamma1,
    gamma2,
    dt_psi,
    d_psi,
    d_phi,
    worldtube_vars,
    dim,
):
    return inverse_spatial_metric
