# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def spatial_velocity(x, t, mean_velocity, comoving_energy_density):
    return np.asarray(mean_velocity)


def lorentz_factor(x, t, mean_velocity, comoving_energy_density):
    return 1.0 / np.sqrt(1.0 - np.linalg.norm(np.asarray(mean_velocity)) ** 2)


def tildeE(x, t, mean_velocity, comoving_energy_density):
    w_sqr = 1.0 / (1.0 - np.linalg.norm(np.asarray(mean_velocity)) ** 2)
    return comoving_energy_density / 3.0 * (4.0 * w_sqr - 1.0)


def tildeS(x, t, mean_velocity, comoving_energy_density):
    w_sqr = 1.0 / (1.0 - np.linalg.norm(np.asarray(mean_velocity)) ** 2)
    prefactor = 4.0 / 3.0 * comoving_energy_density * w_sqr
    return np.asarray(mean_velocity) * prefactor
