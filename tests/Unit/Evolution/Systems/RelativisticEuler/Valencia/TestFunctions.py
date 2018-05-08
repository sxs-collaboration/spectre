# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing ConservativeFromPrimitive.cpp
def tilde_d(rest_mass_density, specific_internal_energy,
            spatial_velocity_oneform, spatial_velocity_squared, lorentz_factor,
            specific_enthalpy, pressure, sqrt_det_spatial_metric):
    return lorentz_factor * rest_mass_density * sqrt_det_spatial_metric


def tilde_tau(rest_mass_density, specific_internal_energy,
              spatial_velocity_oneform, spatial_velocity_squared,
              lorentz_factor, specific_enthalpy, pressure,
              sqrt_det_spatial_metric):
    return ((pressure * spatial_velocity_squared
             + (lorentz_factor/(1.0 + lorentz_factor) * spatial_velocity_squared
                + specific_internal_energy) * rest_mass_density)
            * lorentz_factor**2 * sqrt_det_spatial_metric)


def tilde_s(rest_mass_density, specific_internal_energy,
            spatial_velocity_oneform, spatial_velocity_squared, lorentz_factor,
            specific_enthalpy, pressure, sqrt_det_spatial_metric):
    return (spatial_velocity_oneform * lorentz_factor**2 * specific_enthalpy
            * rest_mass_density * sqrt_det_spatial_metric)


# End functions for testing ConservativeFromPrimitive.cpp


# Functions for testing Fluxes.cpp
def tilde_d_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                 sqrt_det_spatial_metric, pressure, spatial_velocity):
    return tilde_d * (lapse * spatial_velocity - shift)


def tilde_tau_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                   sqrt_det_spatial_metric, pressure, spatial_velocity):
    return (sqrt_det_spatial_metric * lapse * pressure * spatial_velocity
            + tilde_tau * (lapse * spatial_velocity - shift))


def tilde_s_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                 sqrt_det_spatial_metric, pressure, spatial_velocity):
    result = np.outer(lapse * spatial_velocity - shift, tilde_s)
    result += (sqrt_det_spatial_metric * lapse * pressure
               * np.identity(shift.size))
    return result


# End functions for testing Fluxes.cpp
