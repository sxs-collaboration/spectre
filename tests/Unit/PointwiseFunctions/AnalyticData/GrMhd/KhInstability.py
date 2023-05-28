# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def rest_mass_density(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return (
        strip_density
        if np.absolute(x[1] - strip_bimedian_height) < 0.5 * strip_thickness
        else background_density
    )


def electron_fraction(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return 0.1


def velocity(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    dim = x.size
    result = np.zeros(dim)
    result[0] = (
        strip_velocity
        if (np.absolute(x[1] - strip_bimedian_height) < 0.5 * strip_thickness)
        else background_velocity
    )
    strip_lower_bound = strip_bimedian_height - 0.5 * strip_thickness
    strip_upper_bound = strip_bimedian_height + 0.5 * strip_thickness
    result[1] = np.exp(
        -0.5 * ((x[1] - strip_lower_bound) / perturbation_width) ** 2
    ) + np.exp(-0.5 * ((x[1] - strip_upper_bound) / perturbation_width) ** 2)
    result[1] *= perturbation_amplitude * np.sin(4 * np.pi * x[0])
    return result


def specific_internal_energy(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return (
        pressure / (adiabatic_index - 1.0) / strip_density
        if np.absolute(x[1] - strip_bimedian_height) < 0.5 * strip_thickness
        else pressure / (adiabatic_index - 1.0) / background_density
    )


def pressure(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return pressure


def specific_enthalpy(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return 1.0 + adiabatic_index * specific_internal_energy(
        x,
        adiabatic_index,
        strip_bimedian_height,
        strip_thickness,
        strip_density,
        strip_velocity,
        background_density,
        background_velocity,
        pressure,
        perturbation_amplitude,
        perturbation_width,
        magnetic_field,
    )


def lorentz_factor(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    v = velocity(
        x,
        adiabatic_index,
        strip_bimedian_height,
        strip_thickness,
        strip_density,
        strip_velocity,
        background_density,
        background_velocity,
        pressure,
        perturbation_amplitude,
        perturbation_width,
        magnetic_field,
    )
    return 1.0 / np.sqrt(1.0 - np.dot(v, v))


def magnetic_field(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return np.array(magnetic_field)


def divergence_cleaning_field(
    x,
    adiabatic_index,
    strip_bimedian_height,
    strip_thickness,
    strip_density,
    strip_velocity,
    background_density,
    background_velocity,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return 0.0
