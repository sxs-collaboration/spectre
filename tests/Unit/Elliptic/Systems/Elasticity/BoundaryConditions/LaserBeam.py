# Distributed under the MIT License.
# See LICENSE.txt for details.

from math import factorial

import numpy as np
from numpy import cos, exp, pi, sin, sqrt
from numpy.polynomial.hermite import hermval

# Modes (n, m, weight) for the Hermite-Gauss test. Must match `hg_modes()` in
# Test_LaserBeam.cpp.
HG_MODES = [(0, 2, 1.0), (2, 0, -1.0), (1, 3, 0.5)]


def unit_normal(theta, phi):
    return np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])


def normal_dot_minus_stress(x, theta, phi, beam_width):
    n = unit_normal(theta, phi)
    r = sqrt(np.linalg.norm(x) ** 2 - np.dot(x, n) ** 2)
    beam_profile = exp(-((r / beam_width) ** 2)) / pi / beam_width**2
    return np.tensordot(-n, beam_profile, axes=0)


def in_plane_coords(x, n):
    # Same basis convention as LaserBeam.cpp
    t = np.array([0.0, 1.0, 0.0]) if abs(n[0]) > 0.9 else np.array([1, 0, 0.0])
    u = t - np.dot(t, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    r = x - np.dot(x, n) * n
    return np.dot(r, u), np.dot(r, v)


def hermite(n, x):
    # Physicists' Hermite polynomial H_n(x)
    return hermval(x, [0.0] * n + [1.0])


def normal_dot_minus_stress_hermite_gauss(x, theta, phi, beam_width):
    n = unit_normal(theta, phi)
    u, v = in_plane_coords(x, n)
    r0 = beam_width
    beam_profile = 0.0
    for mode_n, mode_m, weight in HG_MODES:
        norm = 2 ** (mode_n + mode_m) * factorial(mode_n) * factorial(mode_m)
        beam_profile += (
            weight
            * (hermite(mode_n, u / r0) * hermite(mode_m, v / r0)) ** 2
            / norm
        )
    beam_profile *= exp(-(u**2 + v**2) / r0**2) / (pi * r0**2)
    return np.tensordot(-n, beam_profile, axes=0)


def normal_dot_minus_stress_linearized(x, theta, phi, beam_width):
    return np.zeros(3)
