# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def rest_mass_density(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    return rest_mass_density


def spatial_velocity(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    magnitude_B0 = np.linalg.norm(bkgd_magnetic_field)
    magnitude_B1 = np.linalg.norm(wave_magnetic_field)
    unit_B0 = np.array(bkgd_magnetic_field) / magnitude_B0
    unit_B1 = np.array(wave_magnetic_field) / magnitude_B1
    unit_E = np.cross(unit_B1, unit_B0)
    rho_zero_times_h = rest_mass_density + pressure * (adiabatic_index) / (
        adiabatic_index - 1.0
    )
    aux_speed_b0 = magnitude_B0 / np.sqrt(
        rho_zero_times_h + magnitude_B0**2 + magnitude_B1**2
    )
    aux_speed_b1 = magnitude_B1 * aux_speed_b0 / magnitude_B0
    one_over_speed_denominator = 1.0 / np.sqrt(
        0.5 * (1.0 + np.sqrt(1.0 - 4.0 * aux_speed_b0**2 * aux_speed_b1**2))
    )
    speed = aux_speed_b0 * one_over_speed_denominator
    phase = wavenumber * (np.dot(unit_B0, x) - speed * t)
    fluid_velocity = -aux_speed_b1 * one_over_speed_denominator
    return fluid_velocity * (np.cos(phase) * unit_B1 - np.sin(phase) * unit_E)


def specific_internal_energy(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    return pressure / (rest_mass_density * (adiabatic_index - 1.0))


def pressure(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    return pressure


def specific_enthalpy(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    return 1.0 + adiabatic_index * specific_internal_energy(
        x,
        t,
        wavenumber,
        pressure,
        rest_mass_density,
        electron_fraction,
        adiabatic_index,
        bkgd_magnetic_field,
        wave_magnetic_field,
    )


def lorentz_factor(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    return 1.0 / np.sqrt(
        1.0
        - np.linalg.norm(
            spatial_velocity(
                x,
                t,
                wavenumber,
                pressure,
                rest_mass_density,
                electron_fraction,
                adiabatic_index,
                bkgd_magnetic_field,
                wave_magnetic_field,
            )
        )
        ** 2
    )


def magnetic_field(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    magnitude_B0 = np.linalg.norm(bkgd_magnetic_field)
    magnitude_B1 = np.linalg.norm(wave_magnetic_field)
    unit_B0 = np.array(bkgd_magnetic_field) / magnitude_B0
    unit_B1 = np.array(wave_magnetic_field) / magnitude_B1
    unit_E = np.cross(unit_B1, unit_B0)
    rho_zero_times_h = rest_mass_density + pressure * (adiabatic_index) / (
        adiabatic_index - 1.0
    )
    aux_speed_b0 = magnitude_B0 / np.sqrt(
        rho_zero_times_h + magnitude_B0**2 + magnitude_B1**2
    )
    aux_speed_b1 = magnitude_B1 * aux_speed_b0 / magnitude_B0
    one_over_speed_denominator = 1.0 / np.sqrt(
        0.5 * (1.0 + np.sqrt(1.0 - 4.0 * aux_speed_b0**2 * aux_speed_b1**2))
    )
    speed = aux_speed_b0 * one_over_speed_denominator
    phase = wavenumber * (np.dot(unit_B0, x) - speed * t)
    return np.array(bkgd_magnetic_field) + magnitude_B1 * (
        np.cos(phase) * unit_B1 - np.sin(phase) * unit_E
    )


def divergence_cleaning_field(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    return 0.0


def electron_fraction(
    x,
    t,
    wavenumber,
    pressure,
    rest_mass_density,
    electron_fraction,
    adiabatic_index,
    bkgd_magnetic_field,
    wave_magnetic_field,
):
    return electron_fraction
