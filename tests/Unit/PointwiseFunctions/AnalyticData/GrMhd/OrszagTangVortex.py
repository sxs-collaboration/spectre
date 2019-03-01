# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def rest_mass_density(x):
    return 25. / (36. * np.pi)


def spatial_velocity(x):
    return np.array([
        -1. / 2. * np.sin(2. * np.pi * x[1]),
        1. / 2. * np.sin(2. * np.pi * x[0]), 0.
    ])


def specific_internal_energy(x):
    return (1. / (5. / 3. - 1.) * pressure(x) / rest_mass_density(x))


def pressure(x):
    return 5. / (12. * np.pi)


def lorentz_factor(x):
    v = spatial_velocity(x)
    return 1. / np.sqrt(1. - np.dot(v, v))


def specific_enthalpy(x):
    return 1. + specific_internal_energy(x) \
        + pressure(x) / rest_mass_density(x)


def magnetic_field(x):
    return np.array([
        -1. / np.sqrt(4. * np.pi) * np.sin(2. * np.pi * x[1]),
        1. / np.sqrt(4. * np.pi) * np.sin(4. * np.pi * x[0]), 0.
    ])


def divergence_cleaning_field(x):
    return 0.
