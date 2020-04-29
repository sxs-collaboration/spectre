# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density_cons_flux_impl(momentum_density, energy_density, velocity,
                                pressure):
    return momentum_density


def momentum_density_flux_impl(momentum_density, energy_density, velocity,
                               pressure):
    result = np.outer(momentum_density, velocity)
    result += pressure * np.identity(velocity.size)
    return result


def energy_density_flux_impl(momentum_density, energy_density, velocity,
                             pressure):
    return (energy_density + pressure) * velocity


def mass_density_cons_flux(momentum_density, energy_density, velocity,
                           pressure, first_arg, second_arg, third_arg,
                           fourth_arg):
    return mass_density_cons_flux_impl(momentum_density, energy_density,
                                       velocity, pressure)


def momentum_density_flux(momentum_density, energy_density, velocity, pressure,
                          first_arg, second_arg, third_arg, fourth_arg):
    return momentum_density_flux_impl(momentum_density, energy_density,
                                      velocity, pressure)


def energy_density_flux(momentum_density, energy_density, velocity, pressure,
                        first_arg, second_arg, third_arg, fourth_arg):
    return energy_density_flux_impl(momentum_density, energy_density, velocity,
                                    pressure)


def source_mass_density_cons_impl(first_arg, second_arg, third_arg,
                                  fourth_arg):
    return np.exp(first_arg)


def source_momentum_density_impl(first_arg, second_arg, third_arg, fourth_arg):
    return (first_arg - 1.5 * third_arg) * second_arg


def source_energy_density_impl(first_arg, second_arg, third_arg, fourth_arg):
    return np.dot(second_arg, fourth_arg) + 3.0 * third_arg


def minus_one_mass_density_impl(first_arg, second_arg, third_arg, fourth_arg):
    return 0.0 * first_arg - 1.0


def source_mass_density_cons(momentum_density, energy_density, velocity,
                             pressure, first_arg, second_arg, third_arg,
                             fourth_arg):
    return source_mass_density_cons_impl(first_arg, second_arg, third_arg,
                                         fourth_arg)


def source_momentum_density(momentum_density, energy_density, velocity,
                            pressure, first_arg, second_arg, third_arg,
                            fourth_arg):
    return source_momentum_density_impl(first_arg, second_arg, third_arg,
                                        fourth_arg)


def source_energy_density(momentum_density, energy_density, velocity, pressure,
                          first_arg, second_arg, third_arg, fourth_arg):
    return source_energy_density_impl(first_arg, second_arg, third_arg,
                                      fourth_arg)


def minus_one_mass_density(momentum_density, energy_density, velocity,
                           pressure, first_arg, second_arg, third_arg,
                           fourth_arg):
    return minus_one_mass_density_impl(first_arg, second_arg, third_arg,
                                       fourth_arg)
