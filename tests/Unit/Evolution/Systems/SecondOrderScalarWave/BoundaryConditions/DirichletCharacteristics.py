# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from Evolution.Systems.SecondOrderScalarWave import Characteristics

_amplitude = 0.9
_width = 0.6
_profile_center = 0.0
_center = np.asarray([1.1, 0.1, -0.9])
_wave_vector = np.asarray([0.1, 1.1, 2.1])

_moving_mesh_error = (
    "DirichletCharacteristics does not support moving meshes for now."
)


def _u(coords, time):
    dim = len(coords)
    k = _wave_vector[:dim]
    omega = np.sqrt(k.dot(k))
    return k.dot(coords - _center[:dim]) - omega * time


def _dprofile(u):
    return (
        (-2.0 * _amplitude / _width**2)
        * (u - _profile_center)
        * np.exp(-((u - _profile_center) ** 2) / _width**2)
    )


def _analytic_pi(coords, time):
    dim = len(coords)
    k = _wave_vector[:dim]
    omega = np.sqrt(k.dot(k))
    return omega * _dprofile(_u(coords, time))


def _analytic_phi(coords, time):
    dim = len(coords)
    return _wave_vector[:dim] * _dprofile(_u(coords, time))


def _exterior_pi_and_phi(
    normal_covector, interior_pi, interior_phi, coords, time, zero_incoming_mode
):
    # The outgoing and zero-speed modes are taken from the interior, the
    # incoming mode from the analytic solution (or zero).
    interior_v_plus = Characteristics.char_field_vplus(
        interior_pi, interior_phi, normal_covector
    )
    interior_v_zero = Characteristics.char_field_vzero(
        interior_pi, interior_phi, normal_covector
    )
    if zero_incoming_mode:
        exterior_v_minus = 0.0
    else:
        exterior_v_minus = Characteristics.char_field_vminus(
            _analytic_pi(coords, time),
            _analytic_phi(coords, time),
            normal_covector,
        )
    exterior_pi = Characteristics.inverse_field_pi(
        interior_v_zero, interior_v_plus, exterior_v_minus, normal_covector
    )
    exterior_phi = Characteristics.inverse_field_phi(
        interior_v_zero, interior_v_plus, exterior_v_minus, normal_covector
    )
    return exterior_pi, exterior_phi


def error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    boundary_psi,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    copy_psi_from_interior,
    zero_incoming_mode,
):
    if face_mesh_velocity is None:
        return None
    return _moving_mesh_error


def ghost_psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    boundary_psi,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    copy_psi_from_interior,
    zero_incoming_mode,
):
    if copy_psi_from_interior:
        return interior_psi
    return boundary_psi


def ghost_pi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    boundary_psi,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    copy_psi_from_interior,
    zero_incoming_mode,
):
    exterior_pi, _ = _exterior_pi_and_phi(
        outward_directed_normal_covector,
        interior_pi,
        interior_phi,
        coords,
        time,
        zero_incoming_mode,
    )
    return exterior_pi


def ghost_phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    boundary_psi,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    copy_psi_from_interior,
    zero_incoming_mode,
):
    _, exterior_phi = _exterior_pi_and_phi(
        outward_directed_normal_covector,
        interior_pi,
        interior_phi,
        coords,
        time,
        zero_incoming_mode,
    )
    return exterior_phi


def dt_boundary_psi_error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    boundary_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    copy_psi_from_interior,
    zero_incoming_mode,
):
    if face_mesh_velocity is None:
        return None
    return _moving_mesh_error


def dt_boundary_psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    boundary_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    copy_psi_from_interior,
    zero_incoming_mode,
):
    # dt Psi = -Pi, evaluated with the exterior Pi.
    exterior_pi, _ = _exterior_pi_and_phi(
        outward_directed_normal_covector,
        interior_pi,
        interior_phi,
        coords,
        time,
        zero_incoming_mode,
    )
    return -exterior_pi
