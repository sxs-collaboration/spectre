# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(face_mesh_velocity, outward_directed_normal_covector, coords,
          interior_gamma2, time, dim):
    return None


def _gauss_amplitude():
    return 0.9


def _gauss_width():
    return 0.6


def _gauss_center():
    return 0.0


def _center(dim):
    center = [1.1]
    if dim > 1:
        center.append(0.1)
        if dim > 2:
            center.append(-0.9)
    return np.asarray(center)


def _wave_vector(dim):
    wave_vector = [0.1]
    if dim > 1:
        wave_vector.append(1.1)
        if dim > 2:
            wave_vector.append(2.1)
    return np.asarray(wave_vector)


def _omega(dim):
    wave_vector = _wave_vector(dim)
    return np.sqrt(wave_vector.dot(wave_vector))


def _1d_u(coords, time, dim):
    result = -_omega(dim) * time
    for i in range(dim):
        result += _wave_vector(dim)[i] * (coords[i] - _center(dim)[i])
    return result


def _profile(u):
    return _gauss_amplitude() * np.exp(
        -(u - _gauss_center())**2 / _gauss_width()**2)


def _first_deriv(u):
    return (-2.0 * _gauss_amplitude() /
            _gauss_width()**2) * (u - _gauss_center()) * np.exp(
                -(u - _gauss_center())**2 / _gauss_width()**2)


def pi(face_mesh_velocity, outward_directed_normal_covector, coords,
       interior_gamma2, time, dim):
    return _omega(dim) * _first_deriv(_1d_u(coords, time, dim))


def phi(face_mesh_velocity, outward_directed_normal_covector, coords,
        interior_gamma2, time, dim):
    result = np.empty([dim])
    du = _first_deriv(_1d_u(coords, time, dim))
    for i in range(dim):
        result[i] = _wave_vector(dim)[i] * du
    return result


def psi(face_mesh_velocity, outward_directed_normal_covector, coords,
        interior_gamma2, time, dim):
    return _profile(_1d_u(coords, time, dim))


def constraint_gamma2(face_mesh_velocity, outward_directed_normal_covector,
                      coords, interior_gamma2, time, dim):
    assert interior_gamma2 >= 0.0  # make sure random gamma_2 is positive
    return interior_gamma2
