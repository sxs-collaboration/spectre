# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


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
