# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def call_operator(coords, amplitude, wavenumber, phase):
    return amplitude * np.sin(wavenumber * coords + phase)[0]


def first_deriv(coords, amplitude, wavenumber, phase):
    return amplitude * wavenumber * np.cos(wavenumber * coords + phase)


def second_deriv(coords, amplitude, wavenumber, phase):
    return np.array(
        [
            -amplitude
            * np.square(wavenumber)
            * np.sin(wavenumber * coords + phase)
        ]
    )


def third_deriv(coords, amplitude, wavenumber, phase):
    return np.array(
        [
            [
                -amplitude
                * wavenumber
                * np.square(wavenumber)
                * np.cos(wavenumber * coords + phase)
            ]
        ]
    )
