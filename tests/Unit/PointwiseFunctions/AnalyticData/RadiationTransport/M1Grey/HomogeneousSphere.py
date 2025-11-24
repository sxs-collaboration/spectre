# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def m1_tildeE(
    x, radius, emissivity_and_opacity, outer_opacity, boundary_roundess
):
    radii = np.linalg.norm(np.asarray(x))
    normalized_radii = (radii - radius) / -boundary_roundess

    energy_difference = 1.0 - 1.0e-12
    energy_sum = 1.0 + 1.0e-12

    e_tilde = (
        energy_difference / np.pi * np.arctan(normalized_radii)
        + 0.5 * energy_sum
    )

    return e_tilde


def m1_tildeS(
    x, radius, emissivity_and_opacity, outer_opacity, boundary_roundess
):
    return np.asarray(x) * 0.0
