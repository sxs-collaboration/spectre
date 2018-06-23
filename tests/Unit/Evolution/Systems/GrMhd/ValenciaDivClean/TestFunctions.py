# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing ConservativeFromPrimitive.cpp
def tilde_d(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
            lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
            spatial_metric, divergence_cleaning_field):
    return lorentz_factor * rest_mass_density * sqrt_det_spatial_metric


def tilde_tau(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
              lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
              spatial_metric, divergence_cleaning_field):
    bsq = np.einsum("ab, ab", spatial_metric,
                    np.outer(magnetic_field, magnetic_field))
    vsq = np.einsum("ab, ab", spatial_metric,
                    np.outer(spatial_velocity, spatial_velocity))
    b_dot_v = np.einsum("ab, ab", spatial_metric,
                        np.outer(magnetic_field, spatial_velocity))
    return ((rest_mass_density * specific_enthalpy * lorentz_factor**2 -
             pressure - rest_mass_density * lorentz_factor + 0.5 * bsq *
             (1.0 + vsq) - 0.5 * b_dot_v) * sqrt_det_spatial_metric)


def tilde_s(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
            lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
            spatial_metric, divergence_cleaning_field):
    spatial_velocity_one_form = np.einsum("a, ia", spatial_velocity,
                                          spatial_metric)
    magnetic_field_one_form = np.einsum("a, ia", magnetic_field,
                                        spatial_metric)
    bsq = np.einsum("ab, ab", spatial_metric,
                    np.outer(magnetic_field, magnetic_field))
    b_dot_v = np.einsum("ab, ab", spatial_metric,
                        np.outer(magnetic_field, spatial_velocity))
    return ((spatial_velocity_one_form *
             (lorentz_factor**2 * specific_enthalpy * rest_mass_density + bsq)
             - magnetic_field_one_form * b_dot_v) * sqrt_det_spatial_metric)


def tilde_b(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
            lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
            spatial_metric, divergence_cleaning_field):
    return sqrt_det_spatial_metric * magnetic_field


def tilde_phi(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
              lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
              spatial_metric, divergence_cleaning_field):
    return sqrt_det_spatial_metric * divergence_cleaning_field


# End functions for testing ConservativeFromPrimitive.cpp
