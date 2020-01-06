# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from PointwiseFunctions.AnalyticSolutions.NewtonianEuler.IsentropicVortex \
    import (deriv_of_perturbation_profile, mass_density, velocity,
            specific_internal_energy, pressure)


def source_mass_density_cons(x, t, adiabatic_index, perturbation_amplitude,
                             vortex_center, vortex_mean_velocity,
                             vortex_strength):
    return (perturbation_amplitude * mass_density(
        x, t, adiabatic_index, vortex_center, vortex_mean_velocity,
        vortex_strength, perturbation_amplitude) *
            deriv_of_perturbation_profile(x[2]))


def source_momentum_density(x, t, adiabatic_index, perturbation_amplitude,
                            vortex_center, vortex_mean_velocity,
                            vortex_strength):
    result = (perturbation_amplitude * velocity(
        x, t, adiabatic_index, vortex_center, vortex_mean_velocity,
        vortex_strength, perturbation_amplitude) * mass_density(
            x, t, adiabatic_index, vortex_center, vortex_mean_velocity,
            vortex_strength, perturbation_amplitude) *
              deriv_of_perturbation_profile(x[2]))
    result[2] *= 2.0
    return result


def source_energy_density(x, t, adiabatic_index, perturbation_amplitude,
                          vortex_center, vortex_mean_velocity,
                          vortex_strength):
    dens = mass_density(x, t, adiabatic_index, vortex_center,
                        vortex_mean_velocity, vortex_strength,
                        perturbation_amplitude)
    veloc = velocity(x, t, adiabatic_index, vortex_center,
                     vortex_mean_velocity, vortex_strength,
                     perturbation_amplitude)
    return (
        perturbation_amplitude *
        (dens * (0.5 * np.dot(veloc, veloc) + specific_internal_energy(
            x, t, adiabatic_index, vortex_center, vortex_mean_velocity,
            vortex_strength, perturbation_amplitude)) +
         pressure(x, t, adiabatic_index, vortex_center, vortex_mean_velocity,
                  vortex_strength, perturbation_amplitude) + dens *
         (veloc[2])**2) * deriv_of_perturbation_profile(x[2]))
