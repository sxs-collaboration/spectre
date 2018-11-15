# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from GeneralRelativity import TestFunctions as gr


# Functions for testing BondiHoyleAccretion.cpp
def bondi_hoyle_rest_mass_density(x, bh_mass, bh_dimless_spin,
                                  rest_mass_density, flow_speed,
                                  mag_field_strength, polytropic_constant,
                                  polytropic_exponent):
    return rest_mass_density


def bondi_hoyle_spatial_velocity(x, bh_mass, bh_dimless_spin,
                                 rest_mass_density, flow_speed,
                                 mag_field_strength, polytropic_constant,
                                 polytropic_exponent):
    result = np.zeros(3)
    spin_a = bh_mass * bh_dimless_spin
    a_squared = spin_a**2
    r_squared = gr.ks_coords_r_squared(x, bh_mass, bh_dimless_spin)
    cos_theta = x[2] / np.sqrt(r_squared)
    cos_theta_squared = cos_theta**2
    sigma = r_squared + a_squared * cos_theta_squared

    result[0] = (flow_speed * cos_theta /
                 np.sqrt(1.0 + 2.0 * bh_mass * np.sqrt(r_squared) /
                         sigma))
    result[1] = - flow_speed * np.sqrt(1.0 - cos_theta_squared) / np.sqrt(sigma)

    return gr.ks_coords_cartesian_from_spherical_ks(result, x, bh_mass,
                                                    bh_dimless_spin)


def bondi_hoyle_specific_internal_energy(x, bh_mass, bh_dimless_spin,
                                         rest_mass_density, flow_speed,
                                         mag_field_strength,
                                         polytropic_constant,
                                         polytropic_exponent):
    return (polytropic_constant *
            pow(rest_mass_density, polytropic_exponent - 1.0) /
            (polytropic_exponent - 1.0))


def bondi_hoyle_pressure(x, bh_mass, bh_dimless_spin, rest_mass_density,
                         flow_speed, mag_field_strength, polytropic_constant,
                         polytropic_exponent):
    return polytropic_constant * pow(rest_mass_density, polytropic_exponent)


def bondi_hoyle_magnetic_field(x, bh_mass, bh_dimless_spin,
                               rest_mass_density, flow_speed,
                               mag_field_strength, polytropic_constant,
                               polytropic_exponent):
    result = np.zeros(3)
    spin_a = bh_mass * bh_dimless_spin
    a_squared = spin_a**2
    r_squared = gr.ks_coords_r_squared(x, bh_mass, bh_dimless_spin)
    cos_theta = x[2] / np.sqrt(r_squared)
    cos_theta_squared = cos_theta**2
    two_m_r = 2.0 * bh_mass * np.sqrt(r_squared)
    sigma = r_squared + a_squared * cos_theta_squared

    result[0:] = mag_field_strength / np.sqrt(sigma * (sigma + two_m_r))
    result[0] *= (r_squared - two_m_r + a_squared + two_m_r *
                  (r_squared**2 - a_squared**2) / sigma**2) * cos_theta
    result[1] *= -((np.sqrt(r_squared) +
                    bh_mass * a_squared *
                    (r_squared - a_squared * cos_theta_squared) *
                    (1.0 + cos_theta_squared) / sigma**2) *
                   np.sqrt(1.0 - cos_theta_squared))
    result[2] *= spin_a * (1.0 + two_m_r * (r_squared - a_squared) /
                           sigma**2) * cos_theta
    return gr.ks_coords_cartesian_from_spherical_ks(result, x, bh_mass,
                                                    bh_dimless_spin)


def bondi_hoyle_divergence_cleaning_field(x, bh_mass, bh_dimless_spin,
                                          rest_mass_density, flow_speed,
                                          mag_field_strength,
                                          polytropic_constant,
                                          polytropic_exponent):
    return 0.0


def bondi_hoyle_lorentz_factor(x, bh_mass, bh_dimless_spin,
                               rest_mass_density, flow_speed,
                               mag_field_strength, polytropic_constant,
                               polytropic_exponent):
    return 1.0 / np.sqrt(1.0 - flow_speed**2)


def bondi_hoyle_specific_enthalpy(x, bh_mass, bh_dimless_spin,
                                  rest_mass_density, flow_speed,
                                  mag_field_strength, polytropic_constant,
                                  polytropic_exponent):
    return (1.0 + (polytropic_constant * polytropic_exponent *
                   pow(rest_mass_density, polytropic_exponent - 1.0) /
                   (polytropic_exponent - 1.0)))


# End functions for testing BondiHoyleAccretion.cpp

# Functions for testing CylindricalBlastWave.cpp
def cylindrical_blast_wave_compute_piecewise(x, inner_radius, outer_radius,
                                             inner_value, outer_value):
    radius = np.sqrt(np.square(x[0]) + np.square(x[1]))
    if (radius > outer_radius):
        return outer_value
    elif (radius < inner_radius):
        return inner_value
    else:
        piecewise_scalar = (-1.0 * radius + inner_radius) * \
            np.log(outer_value)
        piecewise_scalar += (radius - outer_radius) * np.log(inner_value)
        piecewise_scalar /= inner_radius - outer_radius
        return np.exp(piecewise_scalar)


def cylindrical_blast_wave_rest_mass_density(x, inner_radius, outer_radius,
                                             inner_density, outer_density,
                                             inner_pressure, outer_pressure,
                                             magnetic_field, adiabatic_index):
    return cylindrical_blast_wave_compute_piecewise(x, inner_radius, outer_radius,
                                                    inner_density, outer_density)


def cylindrical_blast_wave_spatial_velocity(x, inner_radius, outer_radius,
                                            inner_density, outer_density,
                                            inner_pressure, outer_pressure,
                                            magnetic_field, adiabatic_index):
    return np.zeros(3)


def cylindrical_blast_wave_specific_internal_energy(x, inner_radius, outer_radius,
                                                    inner_density, outer_density,
                                                    inner_pressure,
                                                    outer_pressure,
                                                    magnetic_field,
                                                    adiabatic_index):
    return (1.0 / (adiabatic_index - 1.0) *
            cylindrical_blast_wave_pressure(x, inner_radius, outer_radius,
                                            inner_density, outer_density,
                                            inner_pressure, outer_pressure,
                                            magnetic_field, adiabatic_index) /
            cylindrical_blast_wave_rest_mass_density(x, inner_radius,
                                                     outer_radius,
                                                     inner_density,
                                                     outer_density,
                                                     inner_pressure,
                                                     outer_pressure,
                                                     magnetic_field,
                                                     adiabatic_index))


def cylindrical_blast_wave_pressure(x, inner_radius, outer_radius, inner_density,
                                    outer_density, inner_pressure, outer_pressure,
                                    magnetic_field, adiabatic_index):
    return cylindrical_blast_wave_compute_piecewise(x, inner_radius, outer_radius,
                                                    inner_pressure,
                                                    outer_pressure)


def cylindrical_blast_wave_lorentz_factor(x, inner_radius, outer_radius,
                                          inner_density, outer_density,
                                          inner_pressure, outer_pressure,
                                          magnetic_field, adiabatic_index):
    return 1.0


def cylindrical_blast_wave_specific_enthalpy(x, inner_radius, outer_radius,
                                             inner_density, outer_density,
                                             inner_pressure, outer_pressure,
                                             magnetic_field, adiabatic_index):
    return (1.0 + adiabatic_index *
            cylindrical_blast_wave_specific_internal_energy(x, inner_radius,
                                                            outer_radius,
                                                            inner_density,
                                                            outer_density,
                                                            inner_pressure,
                                                            outer_pressure,
                                                            magnetic_field,
                                                            adiabatic_index))


def cylindrical_blast_wave_magnetic_field(x, inner_radius, outer_radius,
                                          inner_density, outer_density,
                                          inner_pressure, outer_pressure,
                                          magnetic_field, adiabatic_index):
    return np.array(magnetic_field)


def cylindrical_blast_wave_divergence_cleaning_field(x, inner_radius,
                                                     outer_radius, inner_density,
                                                     outer_density,
                                                     inner_pressure,
                                                     outer_pressure,
                                                     magnetic_field,
                                                     adiabatic_index):
    return 0.0

# End for testing CylindricalBlastWave.cpp
