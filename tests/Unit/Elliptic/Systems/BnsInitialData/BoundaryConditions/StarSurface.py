# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def star_surface_normal_dot_flux(
    velocity_potential,
    velocity_potential_gradient,
    lapse,
    rotational_shift,
    euler_enthalpy_constnant,
    normal,
):
    return (
        euler_enthalpy_constnant
        / lapse**2
        * np.einsum("i,i", rotational_shift, normal)
    )


def star_surface_normal_dot_flux_linearized(field_correction):
    return 0.0
