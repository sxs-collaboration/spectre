# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing ProductOfSinusoids.cpp
def parse_arguments(x, wave_numbers):
    x, wave_numbers = np.asarray(x), np.asarray(wave_numbers)
    if wave_numbers.ndim == 0:
        x = np.expand_dims(x, axis=0)
        wave_numbers = np.expand_dims(wave_numbers, axis=0)
    assert len(x) == len(wave_numbers), "Requires the same number of coordinates and wave numbers"
    return x, wave_numbers

def phase(x, wave_numbers):
    x, wave_numbers = parse_arguments(x, wave_numbers)
    return np.array([
        wave_numbers[d] * x[d]
        for d in range(len(x))
    ])

def field(x, wave_numbers):
    x, wave_numbers = parse_arguments(x, wave_numbers)
    return np.prod(np.sin(phase(x, wave_numbers)), axis=0)

def auxiliary_field(x, wave_numbers):
    x, wave_numbers = parse_arguments(x, wave_numbers)
    return np.array([
        wave_numbers[d] * np.cos(np.squeeze(phase(x[d], wave_numbers[d]))) * field(np.delete(x, d, axis=0), np.delete(wave_numbers, d, axis=0))
        for d in range(len(x))
    ])

def source(x, wave_numbers):
    x, wave_numbers = parse_arguments(x, wave_numbers)
    return np.sum(wave_numbers**2) * field(x, wave_numbers)

def auxiliary_source(x, wave_numbers):
    x, wave_numbers = parse_arguments(x, wave_numbers)
    return np.zeros(x.shape)
# End functions for testing ProductOfSinusoids.cpp
