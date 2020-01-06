# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import scipy.optimize as opt


# Functions for testing ConstantM1.cpp
def constant_m1_spatial_velocity(x, t, mean_velocity, comoving_energy_density):
    return np.asarray(mean_velocity)


def constant_m1_lorentz_factor(x, t, mean_velocity, comoving_energy_density):
    return 1.0 / np.sqrt(1.0 - np.linalg.norm(np.asarray(mean_velocity))**2)


def constant_m1_tildeE(x, t, mean_velocity, comoving_energy_density):
    w_sqr = 1.0 / (1.0 - np.linalg.norm(np.asarray(mean_velocity))**2)
    return comoving_energy_density / 3. * (4. * w_sqr - 1.)


def constant_m1_tildeS(x, t, mean_velocity, comoving_energy_density):
    w_sqr = 1.0 / (1.0 - np.linalg.norm(np.asarray(mean_velocity))**2)
    prefactor = 4. / 3. * comoving_energy_density * w_sqr
    return np.asarray(mean_velocity) * prefactor


# End Functions for testing ConstantM1.cpp
