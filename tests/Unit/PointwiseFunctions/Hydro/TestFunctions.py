# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions testing the MassWeightedFluidItems


def mass_weighted_internal_energy(tilde_d, specific_internal_energy):
    return tilde_d * specific_internal_energy


def mass_weighted_kinetic_energy(tilde_d, lorentz_factor):
    return tilde_d * (lorentz_factor - 1.0)


def tilde_d_unbound_ut_criterion(
    tilde_d, lorentz_factor, spatial_velocity, spatial_metric, lapse, shift
):
    shift_dot_velocity = np.einsum(
        "ab, ab", spatial_metric, np.outer(spatial_velocity, shift)
    )
    u_t = lorentz_factor * (-lapse + shift_dot_velocity)
    return tilde_d * (u_t < -1.0)


def mass_weighted_coords_none(tilde_d, grid_coords, compute_coords):
    return tilde_d * compute_coords


def mass_weighted_coords_a(tilde_d, grid_coords, compute_coords):
    return tilde_d * compute_coords * np.heaviside(grid_coords, 0)


def mass_weighted_coords_b(tilde_d, grid_coords, compute_coords):
    return tilde_d * compute_coords * np.heaviside(grid_coords * (-1.0), 0)


# Functions testing the MassWeightedFluidItems


def comoving_magnetic_field(
    spatial_velocity,
    magnetic_field,
    magnetic_field_dot_spatial_velocity,
    lorentz_factor,
    shift,
    lapse,
):
    b_0 = lorentz_factor * magnetic_field_dot_spatial_velocity / lapse
    b_i = magnetic_field / lorentz_factor + (
        lapse * b_0 * (spatial_velocity - shift / lapse)
    )

    return np.concatenate([[b_0], b_i])


def comoving_magnetic_field_one_form(
    spatial_velocity_one_form,
    magnetic_field_one_form,
    magnetic_field_dot_spatial_velocity,
    lorentz_factor,
    shift,
    lapse,
):
    b_i = (
        magnetic_field_one_form / lorentz_factor
        + magnetic_field_dot_spatial_velocity
        * lorentz_factor
        * spatial_velocity_one_form
    )
    b_0 = (
        -lapse * lorentz_factor * magnetic_field_dot_spatial_velocity
        + np.einsum("i...,i...", shift, b_i)
    )

    return np.concatenate([[b_0], b_i])


def comoving_magnetic_field_squared(
    magnetic_field_squared, magnetic_field_dot_spatial_velocity, lorentz_factor
):
    return (
        magnetic_field_squared / lorentz_factor**2
        + magnetic_field_dot_spatial_velocity**2
    )


# Functions for testing LorentzFactor.cpp


def lorentz_factor(spatial_velocity, spatial_velocity_one_form):
    return 1.0 / np.sqrt(
        1.0 - np.dot(spatial_velocity.T, spatial_velocity_one_form)
    )


# End functions for testing LorentzFactor.cpp

# Functions for testing MassFlux.cpp


def mass_flux(
    rest_mass_density,
    spatial_velocity,
    lorentz_factor,
    lapse,
    shift,
    sqrt_det_spatial_metric,
):
    return (
        rest_mass_density
        * lorentz_factor
        * sqrt_det_spatial_metric
        * (lapse * spatial_velocity - shift)
    )


# End functions for testing MassFlux.cpp

# Functions for testing SpecificEnthalpy.cpp


def relativistic_specific_enthalpy(
    rest_mass_density, specific_internal_energy, pressure
):
    return pressure / rest_mass_density + 1.0 + specific_internal_energy


# End functions for testing SpecificEnthalpy.cpp
