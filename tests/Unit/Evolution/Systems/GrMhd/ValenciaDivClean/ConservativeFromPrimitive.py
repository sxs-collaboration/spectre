# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def b_dot_v(magnetic_field, spatial_velocity, spatial_metric):
    return np.einsum(
        "ab, ab", spatial_metric, np.outer(magnetic_field, spatial_velocity)
    )


def b_squared(magnetic_field, spatial_metric):
    return np.einsum(
        "ab, ab", spatial_metric, np.outer(magnetic_field, magnetic_field)
    )


def magnetic_field_one_form(magnetic_field, spatial_metric):
    return np.einsum("a, ia", magnetic_field, spatial_metric)


def p_star(pressure, b_dot_v, b_squared, lorentz_factor):
    return pressure + 0.5 * (b_dot_v**2 + b_squared / lorentz_factor**2)


def spatial_velocity_one_form(spatial_velocity, spatial_metric):
    return np.einsum("a, ia", spatial_velocity, spatial_metric)


def vsq(spatial_velocity, spatial_metric):
    return np.einsum(
        "ab, ab", spatial_metric, np.outer(spatial_velocity, spatial_velocity)
    )


def tilde_d(
    rest_mass_density,
    electron_fraction,
    specific_internal_energy,
    pressure,
    spatial_velocity,
    lorentz_factor,
    magnetic_field,
    sqrt_det_spatial_metric,
    spatial_metric,
    divergence_cleaning_field,
):
    return lorentz_factor * rest_mass_density * sqrt_det_spatial_metric


def tilde_ye(
    rest_mass_density,
    electron_fraction,
    specific_internal_energy,
    pressure,
    spatial_velocity,
    lorentz_factor,
    magnetic_field,
    sqrt_det_spatial_metric,
    spatial_metric,
    divergence_cleaning_field,
):
    return (
        lorentz_factor
        * rest_mass_density
        * sqrt_det_spatial_metric
        * electron_fraction
    )


def tilde_tau(
    rest_mass_density,
    electron_fraction,
    specific_internal_energy,
    pressure,
    spatial_velocity,
    lorentz_factor,
    magnetic_field,
    sqrt_det_spatial_metric,
    spatial_metric,
    divergence_cleaning_field,
):
    spatial_velocity_squared = vsq(spatial_velocity, spatial_metric)
    return (
        (
            (
                pressure * spatial_velocity_squared
                + (
                    lorentz_factor
                    / (1.0 + lorentz_factor)
                    * spatial_velocity_squared
                    + specific_internal_energy
                )
                * rest_mass_density
            )
            * lorentz_factor**2
        )
        + 0.5
        * b_squared(magnetic_field, spatial_metric)
        * (1.0 + spatial_velocity_squared)
        - 0.5
        * np.square(b_dot_v(magnetic_field, spatial_velocity, spatial_metric))
    ) * sqrt_det_spatial_metric


def tilde_s(
    rest_mass_density,
    electron_fraction,
    specific_internal_energy,
    pressure,
    spatial_velocity,
    lorentz_factor,
    magnetic_field,
    sqrt_det_spatial_metric,
    spatial_metric,
    divergence_cleaning_field,
):
    return (
        spatial_velocity_one_form(spatial_velocity, spatial_metric)
        * (
            lorentz_factor**2
            * (pressure + rest_mass_density * (1.0 + specific_internal_energy))
            + b_squared(magnetic_field, spatial_metric)
        )
        - magnetic_field_one_form(magnetic_field, spatial_metric)
        * b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    ) * sqrt_det_spatial_metric


def tilde_b(
    rest_mass_density,
    electron_fraction,
    specific_internal_energy,
    pressure,
    spatial_velocity,
    lorentz_factor,
    magnetic_field,
    sqrt_det_spatial_metric,
    spatial_metric,
    divergence_cleaning_field,
):
    return sqrt_det_spatial_metric * magnetic_field


def tilde_phi(
    rest_mass_density,
    electron_fraction,
    specific_internal_energy,
    pressure,
    spatial_velocity,
    lorentz_factor,
    magnetic_field,
    sqrt_det_spatial_metric,
    spatial_metric,
    divergence_cleaning_field,
):
    return sqrt_det_spatial_metric * divergence_cleaning_field
