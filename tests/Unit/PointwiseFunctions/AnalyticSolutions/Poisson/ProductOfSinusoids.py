# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def field(x, wave_numbers):
    x, wave_numbers = np.asarray(x), np.asarray(wave_numbers)
    return np.prod(np.sin(wave_numbers * x))


def field_gradient(x, wave_numbers):
    x, wave_numbers = np.asarray(x), np.asarray(wave_numbers)
    try:
        dim = len(x)
    except TypeError:
        dim = 1
    return wave_numbers * np.cos(wave_numbers * x) * \
        np.array([field(np.delete(x, d), np.delete(wave_numbers, d))
                  for d in range(dim)])


def field_flux(x, wave_numbers):
    return field_gradient(x, wave_numbers)


def source(x, wave_numbers):
    x, wave_numbers = np.asarray(x), np.asarray(wave_numbers)
    return np.sum(wave_numbers**2) * field(x, wave_numbers)
