# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density(coords, initial_radius, inner_mass_density, inner_pressure,
                 outer_mass_density, outer_pressure):
    radius = np.sqrt(np.dot(coords, coords))
    return (inner_mass_density
            if radius <= initial_radius else outer_mass_density)


def pressure(coords, initial_radius, inner_mass_density, inner_pressure,
             outer_mass_density, outer_pressure):
    radius = np.sqrt(np.dot(coords, coords))
    return (inner_pressure if radius <= initial_radius else outer_pressure)


def velocity(coords, initial_radius, inner_mass_density, inner_pressure,
             outer_mass_density, outer_pressure):
    return np.zeros([len(coords)])


def specific_internal_energy(coords, initial_radius, inner_mass_density,
                             inner_pressure, outer_mass_density,
                             outer_pressure):
    adiabatic_index = 1.4
    return pressure(coords, initial_radius, inner_mass_density, inner_pressure,
                    outer_mass_density, outer_pressure) / mass_density(
                        coords, initial_radius, inner_mass_density,
                        inner_pressure, outer_mass_density,
                        outer_pressure) / (adiabatic_index - 1.0)
