# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def rest_mass_density(x):
    return 25.0 / (36.0 * np.pi)


def spatial_velocity(x):
    return np.array(
        [
            -1.0 / 2.0 * np.sin(2.0 * np.pi * x[1]),
            1.0 / 2.0 * np.sin(2.0 * np.pi * x[0]),
            0.0,
        ]
    )


def specific_internal_energy(x):
    return 1.0 / (5.0 / 3.0 - 1.0) * pressure(x) / rest_mass_density(x)


def pressure(x):
    return 5.0 / (12.0 * np.pi)


def lorentz_factor(x):
    v = spatial_velocity(x)
    return 1.0 / np.sqrt(1.0 - np.dot(v, v))


def specific_enthalpy(x):
    return (
        1.0 + specific_internal_energy(x) + pressure(x) / rest_mass_density(x)
    )


def magnetic_field(x):
    return np.array(
        [
            -1.0 / np.sqrt(4.0 * np.pi) * np.sin(2.0 * np.pi * x[1]),
            1.0 / np.sqrt(4.0 * np.pi) * np.sin(4.0 * np.pi * x[0]),
            0.0,
        ]
    )


def divergence_cleaning_field(x):
    return 0.0
