# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


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
