# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing SmoothFlow.cpp
def rest_mass_density(x, t, mean_velocity, wave_vector, pressure,
                     adiabatic_exponent, density_amplitude):
    return 1.0 + (density_amplitude *
                  np.sin(np.dot(np.asarray(wave_vector),
                                np.asarray(x) - np.asarray(mean_velocity) * t)))


def spatial_velocity(x, t, mean_velocity, wave_vector, pressure,
                     adiabatic_exponent, density_amplitude):
    return np.asarray(mean_velocity)


def specific_internal_energy(x, t, mean_velocity, wave_vector, pressure,
                             adiabatic_exponent, density_amplitude):
    return (pressure /
            ((adiabatic_exponent - 1.0) *
             rest_mass_density(x, t, mean_velocity, wave_vector, pressure,
                               adiabatic_exponent, density_amplitude)))


def pressure(x, t, mean_velocity, wave_vector, pressure,
             adiabatic_exponent, density_amplitude):
    return pressure


def magnetic_field(x, t, mean_velocity, wave_vector, pressure,
             adiabatic_exponent, density_amplitude):
    return np.array([0.0,0.0,0.0])


def dt_rest_mass_density(x, t, mean_velocity, wave_vector, pressure,
                     adiabatic_exponent, density_amplitude):
    return (-density_amplitude *
           np.dot(np.asarray(wave_vector),np.asarray(mean_velocity)) *
                  np.cos(np.dot(np.asarray(wave_vector),
                                np.asarray(x) - np.asarray(mean_velocity) * t)))



def dt_spatial_velocity(x, t, mean_velocity, wave_vector, pressure,
                     adiabatic_exponent, density_amplitude):
    return np.array([0.0,0.0,0.0])


def dt_specific_internal_energy(x, t, mean_velocity, wave_vector, pressure,
                             adiabatic_exponent, density_amplitude):
    return (-pressure /
            ((adiabatic_exponent - 1.0) * rest_mass_density(x, t, mean_velocity, wave_vector, pressure,
                               adiabatic_exponent, density_amplitude)**2 ) *
             dt_rest_mass_density(x, t, mean_velocity, wave_vector, pressure,
                               adiabatic_exponent, density_amplitude))


def dt_pressure(x, t, mean_velocity, wave_vector, pressure,
             adiabatic_exponent, density_amplitude):
    return 0.0


def dt_magnetic_field(x, t, mean_velocity, wave_vector, pressure,
             adiabatic_exponent, density_amplitude):
    return np.array([0.0,0.0,0.0])


# End functions for testing SmoothFlow.cpp
