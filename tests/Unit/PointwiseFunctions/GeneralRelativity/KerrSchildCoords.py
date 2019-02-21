# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def r_coord_squared(coords, bh_mass, bh_dimless_spin):
    a_squared = (bh_mass * bh_dimless_spin)**2
    temp = 0.5 * (coords[0] * coords[0] + coords[1] * coords[1] +
                  coords[2] * coords[2] - a_squared)
    return temp + np.sqrt(temp * temp + a_squared * coords[2] * coords[2])


def jacobian(coords, bh_mass, bh_dimless_spin):
    result = np.zeros((3, 3))
    spin_a = bh_mass * bh_dimless_spin
    a_squared = spin_a**2
    r_squared = r_coord_squared(coords, bh_mass, bh_dimless_spin)
    r = np.sqrt(r_squared)
    sin_theta = np.sqrt((coords[0]**2 + coords[1]**2) /
                        (r_squared + a_squared))
    cos_theta = coords[2] / r
    inv_denom = 1.0 / np.sqrt((coords[0]**2 + coords[1]**2) *
                              (r_squared + a_squared))
    sin_phi = (coords[1] * r - spin_a * coords[0]) * inv_denom
    cos_phi = (coords[0] * r + spin_a * coords[1]) * inv_denom

    result[0, 0] = sin_theta * cos_phi
    result[0, 1] = (r * cos_phi - spin_a * sin_phi) * cos_theta
    result[0, 2] = -(r * sin_phi + spin_a * cos_phi) * sin_theta
    result[1, 0] = sin_theta * sin_phi
    result[1, 1] = (r * sin_phi + spin_a * cos_phi) * cos_theta
    result[1, 2] = (r * cos_phi - spin_a * sin_phi) * sin_theta
    result[2, 0] = cos_theta
    result[2, 1] = -r * sin_theta
    result[2, 2] = 0.0
    return result


def cartesian_from_spherical_ks(vector, cartesian_coords, bh_mass,
                                bh_dimless_spin):
    return np.einsum("ij,j",
                     jacobian(cartesian_coords, bh_mass, bh_dimless_spin),
                     vector)
