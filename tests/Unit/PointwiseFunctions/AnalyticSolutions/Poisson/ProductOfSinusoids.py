# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def field(x, wave_numbers, complex_phase):
    x, wave_numbers = np.asarray(x), np.asarray(wave_numbers)
    result = np.prod(np.sin(wave_numbers * x))
    if complex_phase != 0.0:
        result *= np.exp(1j * complex_phase)
    return result


def field_gradient(x, wave_numbers, complex_phase):
    x, wave_numbers = np.asarray(x), np.asarray(wave_numbers)
    try:
        dim = len(x)
    except TypeError:
        dim = 1
    return (
        wave_numbers
        * np.cos(wave_numbers * x)
        * np.array(
            [
                field(
                    np.delete(x, d), np.delete(wave_numbers, d), complex_phase
                )
                for d in range(dim)
            ]
        )
    )


def field_flux(x, wave_numbers, complex_phase):
    return field_gradient(x, wave_numbers, complex_phase)


def source(x, wave_numbers, complex_phase):
    x, wave_numbers = np.asarray(x), np.asarray(wave_numbers)
    return np.sum(wave_numbers**2) * field(x, wave_numbers, complex_phase)
