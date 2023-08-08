# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions testing QuadrupoleFormula


def quadrupole_moment(tilde_d, compute_coords):
    xi, xj = np.meshgrid(compute_coords, compute_coords, indexing="ij")
    return tilde_d * xi * xj


def quadrupole_moment_derivative(tilde_d, compute_coords, spatial_velocity):
    xi, xj = np.meshgrid(compute_coords, compute_coords, indexing="ij")
    vi, vj = np.meshgrid(spatial_velocity, spatial_velocity, indexing="ij")
    return tilde_d * (vi * xj + xi * vj)


# End functions for testing QuadrupoleFormula
