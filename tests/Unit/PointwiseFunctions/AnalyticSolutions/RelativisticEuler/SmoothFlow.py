# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def rest_mass_density(x, t, mean_velocity, wave_vector, pressure,
                      adiabatic_index, density_amplitude):
    return (1.0 + (density_amplitude * np.sin(
        np.dot(np.asarray(wave_vector),
               np.asarray(x) - np.asarray(mean_velocity) * t))))


def spatial_velocity(x, t, mean_velocity, wave_vector, pressure,
                     adiabatic_index, density_amplitude):
    return np.asarray(mean_velocity)


def specific_internal_energy(x, t, mean_velocity, wave_vector, pressure,
                             adiabatic_index, density_amplitude):
    return (pressure /
            ((adiabatic_index - 1.0) *
             rest_mass_density(x, t, mean_velocity, wave_vector, pressure,
                               adiabatic_index, density_amplitude)))


def pressure(x, t, mean_velocity, wave_vector, pressure, adiabatic_index,
             density_amplitude):
    return pressure


def specific_enthalpy(x, t, mean_velocity, wave_vector, pressure,
                      adiabatic_index, density_amplitude):
    return 1.0 + adiabatic_index * specific_internal_energy(
        x, t, mean_velocity, wave_vector, pressure, adiabatic_index,
        density_amplitude)


def lorentz_factor(x, t, mean_velocity, wave_vector, pressure, adiabatic_index,
                   density_amplitude):
    return 1.0 / np.sqrt(1.0 - np.linalg.norm(np.asarray(mean_velocity))**2)
