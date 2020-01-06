# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def gravitational_field(x, central_mass_density, polytropic_constant):
    alpha = np.sqrt(0.5 * polytropic_constant / np.pi)
    outer_radius = alpha * np.pi
    mass_scale = 4.0 * np.pi * alpha**3 * central_mass_density
    radius = np.sqrt(np.sum(x**2)) + 1.e-30 * outer_radius
    if (radius < outer_radius):
        xi = radius / alpha
        enclosed_mass = mass_scale * (np.sin(xi) - xi * np.cos(xi))
        return -x * enclosed_mass / radius**3
    else:
        total_mass = mass_scale * np.pi
        return -x * total_mass / radius**3


def mass_density(x, t, central_mass_density, polytropic_constant):
    alpha = np.sqrt(0.5 * polytropic_constant / np.pi)
    outer_radius = alpha * np.pi
    radius = np.sqrt(np.sum(x**2)) + 1.e-30 * outer_radius
    if (radius < outer_radius):
        xi = radius / alpha
        return central_mass_density * np.sin(xi) / xi
    else:
        return 0.0


def velocity(x, t, central_mass_density, polytropic_constant):
    return np.zeros(3)


def specific_internal_energy(x, t, central_mass_density, polytropic_constant):
    rho = mass_density(x, t, central_mass_density, polytropic_constant)
    return polytropic_constant * rho


def pressure(x, t, central_mass_density, polytropic_constant):
    rho = mass_density(x, t, central_mass_density, polytropic_constant)
    return polytropic_constant * rho**2
