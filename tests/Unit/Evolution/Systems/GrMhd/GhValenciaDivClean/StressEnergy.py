# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def four_velocity_one_form(rest_mass_density, specific_enthalpy,
                           spatial_velocity_one_form, magnetic_field_one_form,
                           magnetic_field_squared,
                           magnetic_field_dot_spatial_velocity, lorentz_factor,
                           one_over_w_squared, pressure, spacetime_metric,
                           shift, lapse):
    spatial_four_velocity = lorentz_factor * spatial_velocity_one_form
    return np.insert(
        spatial_four_velocity, 0,
        np.einsum("a,a", shift, spatial_four_velocity) -
        lapse * lorentz_factor)


def comoving_magnetic_field_one_form(
    rest_mass_density, specific_enthalpy, spatial_velocity_one_form,
    magnetic_field_one_form, magnetic_field_squared,
    magnetic_field_dot_spatial_velocity, lorentz_factor, one_over_w_squared,
    pressure, spacetime_metric, shift, lapse):
    spatial_comoving_magnetic_field_one_form = (
        magnetic_field_one_form / lorentz_factor + lorentz_factor *
        magnetic_field_dot_spatial_velocity * spatial_velocity_one_form)

    return np.insert(
        spatial_comoving_magnetic_field_one_form, 0,
        np.einsum("a,a", shift, spatial_comoving_magnetic_field_one_form) -
        lapse * lorentz_factor * magnetic_field_dot_spatial_velocity)


def trace_reversed_stress_energy(
    rest_mass_density, specific_enthalpy, spatial_velocity_one_form,
    magnetic_field_one_form, magnetic_field_squared,
    magnetic_field_dot_spatial_velocity, lorentz_factor, one_over_w_squared,
    pressure, spacetime_metric, shift, lapse):
    local_four_velocity_one_form = four_velocity_one_form(
        rest_mass_density, specific_enthalpy, spatial_velocity_one_form,
        magnetic_field_one_form, magnetic_field_squared,
        magnetic_field_dot_spatial_velocity, lorentz_factor,
        one_over_w_squared, pressure, spacetime_metric, shift, lapse)
    local_comoving_magnetic_field_one_form =\
      comoving_magnetic_field_one_form(
        rest_mass_density, specific_enthalpy, spatial_velocity_one_form,
        magnetic_field_one_form, magnetic_field_squared,
        magnetic_field_dot_spatial_velocity, lorentz_factor, one_over_w_squared,
        pressure, spacetime_metric, shift, lapse)
    modified_enthalpy_times_rest_mass = (
        rest_mass_density * specific_enthalpy +
        magnetic_field_squared * one_over_w_squared +
        magnetic_field_dot_spatial_velocity**2)

    return (
        modified_enthalpy_times_rest_mass *
        np.outer(local_four_velocity_one_form, local_four_velocity_one_form) +
        (0.5 * modified_enthalpy_times_rest_mass - pressure) * spacetime_metric
        - np.outer(local_comoving_magnetic_field_one_form,
                   local_comoving_magnetic_field_one_form))


def add_stress_energy_term_to_dt_pi(trace_reversed_stress_energy, lapse):
    return 0.1234 - 16.0 * np.pi * lapse * trace_reversed_stress_energy
