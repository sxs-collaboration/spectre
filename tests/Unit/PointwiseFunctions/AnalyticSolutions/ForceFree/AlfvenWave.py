# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def solution_wave_frame(coords_prime):
    electric_field = coords_prime * 0.0
    magnetic_field = coords_prime * 0.0

    x = coords_prime[0]

    magnetic_field[0] = 1.0
    magnetic_field[1] = 1.0
    magnetic_field[2] = np.where(
        x > -0.1, 1.0 + 0.15 * (1.0 + np.sin(5 * np.pi * x)), 1.0
    )
    magnetic_field[2] = np.where(x > 0.1, 1.3, magnetic_field[2])

    electric_field[0] = -magnetic_field[2]
    electric_field[2] = 1.0

    return (electric_field, magnetic_field)


def lorentz_transform_em_field(v, em_fields_tuple):
    # transforms electric and magnetic field to their values in the frame
    # moving with the speed (v, 0, 0) with respect to the original frame
    electric_field, magnetic_field = em_fields_tuple

    electric_field_prime = electric_field * 1.0
    magnetic_field_prime = magnetic_field * 1.0

    lorentz_factor = 1.0 / np.sqrt(1 - v**2)

    electric_field_prime[1] = lorentz_factor * (
        electric_field[1] - v * magnetic_field[2]
    )
    electric_field_prime[2] = lorentz_factor * (
        electric_field[2] + v * magnetic_field[1]
    )

    magnetic_field_prime[1] = lorentz_factor * (
        magnetic_field[1] + v * electric_field[2]
    )
    magnetic_field_prime[2] = lorentz_factor * (
        magnetic_field[2] - v * electric_field[1]
    )

    return (electric_field_prime, magnetic_field_prime)


def TildeE(x, t, wave_speed):
    lorentz_factor = 1.0 / np.sqrt(1 - wave_speed**2)
    x0 = x * 1.0
    x0[0] -= wave_speed * t

    return lorentz_transform_em_field(
        -wave_speed, solution_wave_frame(lorentz_factor * x0)
    )[0]


def TildeB(x, t, wave_speed):
    lorentz_factor = 1.0 / np.sqrt(1 - wave_speed**2)
    x0 = x * 1.0
    x0[0] -= wave_speed * t
    return lorentz_transform_em_field(
        -wave_speed, solution_wave_frame(lorentz_factor * x0)
    )[1]


def TildePsi(x, t, wave_speed):
    return x[0] * 0


def TildePhi(x, t, wave_speed):
    return x[0] * 0


def solution_charge_wave_frame(x):
    q = np.where(x < -0.1, 0.0, -0.75 * np.pi * np.cos(5.0 * np.pi * x))
    q = np.where(x > 0.1, 0.0, q)
    return q


def TildeQ(x, t, wave_speed):
    lorentz_factor = 1.0 / np.sqrt(1 - wave_speed**2)
    return lorentz_factor * solution_charge_wave_frame(
        lorentz_factor * (x[0] - wave_speed * t)
    )
