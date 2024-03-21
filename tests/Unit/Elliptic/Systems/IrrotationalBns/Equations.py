# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def flat_potential_fluxes(auxilliary_velocity):
    return auxilliary_velocity


def curved_potential_fluxes(inv_spatial_metric, auxiliary_velocity):
    return np.einsum("ij,j", inv_spatial_metric, auxiliary_velocity)


def add_flat_cartesian_sources(
    upper_auxiliary_velocity, log_deriv_lapse_over_specific_enthalpy
):
    return np.einsum(
        "i, i", upper_auxiliary_velocity, log_deriv_lapse_over_specific_enthalpy
    )


def add_curved_potential_sources(
    log_deriv_lapse_over_specific_enthalpy,
    christoffel_contracted,
    potential_flux,
):
    return -np.einsum(
        "i,i",
        christoffel_contracted - log_deriv_lapse_over_specific_enthalpy,
        potential_flux,
    )


def auxiliary_fluxes(velocity_potential, rotational_shift_stress):
    return velocity_potential * (np.eye(3) - rotational_shift_stress / 2)


def add_auxiliary_sources_without_flux_christoffels(
    velocity_potential,
    div_rotational_shift_stress,
    fixed_sources,
):
    return (
        -velocity_potential * div_rotational_shift_stress / 2.0 - fixed_sources
    )


def add_auxiliary_source_flux_christoffels(
    velocity_potential,
    christoffel_contracted,
    christoffel,
    rotational_shift_stress,
):
    return -velocity_potential * (
        np.einsum("kji, jk", christoffel, rotational_shift_stress)
        - np.einsum("i, ij", christoffel_contracted, rotational_shift_stress)
    )
