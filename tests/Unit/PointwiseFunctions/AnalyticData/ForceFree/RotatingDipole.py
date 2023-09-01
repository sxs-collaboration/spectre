# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from scipy.spatial.transform import Rotation

A0 = 1.1
varpi0 = 0.3
delta = 0.07
Omega = 0.123
alpha = 0.456


def TildeE(
    x,
    vector_potential_amplitude,
    varpi0,
    delta,
    angular_velocity,
    tilt_angle,
):
    return x * 0.0


def TildeB(
    x,
    vector_potential_amplitude,
    varpi0,
    delta,
    angular_velocity,
    tilt_angle,
):
    tilde_b = x * 0.0

    r = np.sqrt(np.einsum("a, a", x, x))

    tilt = Rotation.from_rotvec(alpha * np.array([0, 1, 0]))
    x_prime = tilt.inv().apply(x)
    xp, yp, zp = x_prime

    tilde_b[0] = 3.0 * xp * zp
    tilde_b[1] = 3.0 * yp * zp
    tilde_b[2] = 3.0 * zp**2 - r**2 + 2.0 * delta**2
    tilde_b = tilde_b / (r**2 + delta**2) ** (5 / 2)

    return tilt.apply(tilde_b)


def TildePsi(
    x,
    vector_potential_amplitude,
    varpi0,
    delta,
    angular_velocity,
    tilt_angle,
):
    return 0.0


def TildePhi(
    x,
    vector_potential_amplitude,
    varpi0,
    delta,
    angular_velocity,
    tilt_angle,
):
    return 0.0


def TildeQ(
    x,
    vector_potential_amplitude,
    varpi0,
    delta,
    angular_velocity,
    tilt_angle,
):
    return 0.0


def InteriorMask(x):
    r_squared = np.einsum("a, a", x, x)

    if r_squared < 1.0:
        return -1.0
    else:
        return 1.0
