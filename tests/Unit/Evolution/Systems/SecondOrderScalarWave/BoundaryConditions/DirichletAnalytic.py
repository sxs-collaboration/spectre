# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

_amplitude = 0.9
_width = 0.6
_profile_center = 0.0
_center = np.asarray([1.1, 0.1, -0.9])
_wave_vector = np.asarray([0.1, 1.1, 2.1])


def _u(coords, time):
    dim = len(coords)
    k = _wave_vector[:dim]
    omega = np.sqrt(k.dot(k))
    return k.dot(coords - _center[:dim]) - omega * time


def _profile(u):
    return _amplitude * np.exp(-((u - _profile_center) ** 2) / _width**2)


def _dprofile(u):
    return (
        (-2.0 * _amplitude / _width**2)
        * (u - _profile_center)
        * np.exp(-((u - _profile_center) ** 2) / _width**2)
    )


def error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
):
    return None


def psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
):
    return _profile(_u(coords, time))


def pi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
):
    dim = len(coords)
    k = _wave_vector[:dim]
    omega = np.sqrt(k.dot(k))
    return omega * _dprofile(_u(coords, time))


def phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
):
    dim = len(coords)
    return _wave_vector[:dim] * _dprofile(_u(coords, time))
