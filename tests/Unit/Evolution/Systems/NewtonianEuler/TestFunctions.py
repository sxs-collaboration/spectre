# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing Characteristics.cpp
def characteristic_speeds(velocity, sound_speed, normal):
    normal_velocity = np.dot(velocity, normal)
    result = [normal_velocity - sound_speed]
    for i in range(0, velocity.size):
        result.append(normal_velocity)
    result.append(normal_velocity + sound_speed)
    return result


# End functions for testing Characteristics.cpp


# Functions for testing ConservativeFromPrimitive.cpp
def mass_density_cons(mass_density, velocity, specific_internal_energy):
    return mass_density


def momentum_density(mass_density, velocity, specific_internal_energy):
    return mass_density * velocity


def energy_density(mass_density, velocity, specific_internal_energy):
    return (0.5 * mass_density * np.dot(velocity, velocity) +
            mass_density * specific_internal_energy)


# End functions for testing ConservativeFromPrimitive.cpp


# Functions for testing Fluxes.cpp
def mass_density_cons_flux(momentum_density, energy_density,
                           velocity, pressure):
    return momentum_density


def momentum_density_flux(momentum_density, energy_density, velocity, pressure):
    result = np.outer(momentum_density, velocity)
    result += pressure * np.identity(velocity.size)
    return result


def energy_density_flux(momentum_density, energy_density, velocity, pressure):
    return (energy_density + pressure) * velocity


# End functions for testing Fluxes.cpp


# Functions for testing PrimitiveFromConservative.cpp
def mass_density(mass_density_cons, momentum_density, energy_density):
    return mass_density_cons


def velocity(mass_density_cons, momentum_density, energy_density):
    return (momentum_density / mass_density_cons)


def specific_internal_energy(mass_density_cons, momentum_density,
                             energy_density):
    veloc = velocity(mass_density_cons, momentum_density, energy_density)
    return (energy_density / mass_density_cons - 0.5 * np.dot(veloc, veloc))


# Hard-coded for PolytropicFluid (representative ThermoDim = 1 case)
def pressure_1d(mass_density_cons, momentum_density, energy_density):
    return 1.4 * np.power(mass_density_cons, 5.0 / 3.0)


# Hard-coded for IdealFluid (representative ThermoDim = 2 case)
def pressure_2d(mass_density_cons, momentum_density, energy_density):
    return ((2.0 / 3.0) * mass_density_cons *
            specific_internal_energy(mass_density_cons, momentum_density,
                                     energy_density))


# End functions for testing PrimitiveFromConservative.cpp
