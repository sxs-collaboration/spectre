# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing Characteristics.cpp
def characteristic_speeds(velocity, sound_speed_squared, normal):
    normal_velocity = np.dot(velocity, normal)
    sound_speed = np.sqrt(sound_speed_squared)
    result = [normal_velocity - sound_speed]
    for i in range(0, velocity.size):
        result.append(normal_velocity)
    result.append(normal_velocity + sound_speed)
    return result


# End functions for testing Characteristics.cpp


# Functions for testing ConservativeFromPrimitive.cpp
def momentum_density(mass_density, velocity, specific_internal_energy):
    return mass_density * velocity


def energy_density(mass_density, velocity, specific_internal_energy):
    return (0.5 * mass_density * np.dot(velocity, velocity) +
            mass_density * specific_internal_energy)


# End functions for testing ConservativeFromPrimitive.cpp


# Functions for testing Fluxes.cpp
def mass_density_flux(momentum_density, energy_density, velocity, pressure):
    return momentum_density


def momentum_density_flux(momentum_density, energy_density, velocity, pressure):
    result = np.outer(momentum_density, velocity)
    result += pressure * np.identity(velocity.size)
    return result


def energy_density_flux(momentum_density, energy_density, velocity, pressure):
    return (energy_density + pressure) * velocity


# End functions for testing Fluxes.cpp


# Functions for testing PrimitiveFromConservative.cpp
def velocity(mass_density, momentum_density, energy_density = None):
    return (momentum_density / mass_density)


def specific_internal_energy(mass_density, momentum_density, energy_density):
    veloc = velocity(mass_density, momentum_density, energy_density)
    return (energy_density / mass_density - 0.5 * np.dot(veloc, veloc))


# End functions for testing PrimitiveFromConservative.cpp
