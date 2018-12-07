# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import scipy.optimize as opt

# Functions for testing SmoothFlow.cpp
def smooth_flow_rest_mass_density(x, t, mean_velocity, wave_vector, pressure,
                                  adiabatic_index, density_amplitude):
    return (1.0 + (density_amplitude * np.sin(
        np.dot(
            np.asarray(wave_vector),
            np.asarray(x) - np.asarray(mean_velocity) * t))))


def smooth_flow_spatial_velocity(x, t, mean_velocity, wave_vector, pressure,
                                 adiabatic_index, density_amplitude):
    return np.asarray(mean_velocity)


def smooth_flow_specific_internal_energy(x, t, mean_velocity, wave_vector,
                                         pressure, adiabatic_index,
                                         density_amplitude):
    return (
        pressure / ((adiabatic_index - 1.0) * smooth_flow_rest_mass_density(
            x, t, mean_velocity, wave_vector, pressure, adiabatic_index,
            density_amplitude)))


def smooth_flow_pressure(x, t, mean_velocity, wave_vector, pressure,
                         adiabatic_index, density_amplitude):
    return pressure


def smooth_flow_specific_enthalpy(x, t, mean_velocity, wave_vector, pressure,
                                  adiabatic_index, density_amplitude):
    return 1.0 + adiabatic_index * smooth_flow_specific_internal_energy(
        x, t, mean_velocity, wave_vector, pressure, adiabatic_index,
        density_amplitude)


def smooth_flow_lorentz_factor(x, t, mean_velocity, wave_vector, pressure,
                               adiabatic_index, density_amplitude):
    return 1.0 / np.sqrt(1.0 - np.linalg.norm(np.asarray(mean_velocity))**2)


def smooth_flow_magnetic_field(x, t, mean_velocity, wave_vector, pressure,
                               adiabatic_index, density_amplitude):
    return np.array([0.0, 0.0, 0.0])


def smooth_flow_divergence_cleaning_field(x, t, mean_velocity, wave_vector,
                                          pressure, adiabatic_index,
                                          density_amplitude):
    return 0.0


# End functions for testing SmoothFlow.cpp


# Functions for testing AlfvenWave.cpp
def alfven_rest_mass_density(x, t, wavenumber, pressure, rest_mass_density,
                             adiabatic_index, bkgd_magnetic_field,
                             wave_magnetic_field):
    return rest_mass_density


def alfven_spatial_velocity(x, t, wavenumber, pressure, rest_mass_density,
                            adiabatic_index, bkgd_magnetic_field,
                            wave_magnetic_field):
    magnitude_B0 = np.linalg.norm(bkgd_magnetic_field)
    magnitude_B1 = np.linalg.norm(wave_magnetic_field)
    unit_B0 = np.array(bkgd_magnetic_field) / magnitude_B0
    unit_B1 = np.array(wave_magnetic_field) / magnitude_B1
    unit_E = np.cross(unit_B1, unit_B0)
    rho_zero_times_h = \
     (rest_mass_density + pressure *
      (adiabatic_index)/(adiabatic_index - 1.0))
    aux_speed_b0 = magnitude_B0 / \
     np.sqrt(rho_zero_times_h + magnitude_B0 ** 2 + \
     magnitude_B1 **2)
    aux_speed_b1 = magnitude_B1 * aux_speed_b0 / magnitude_B0
    one_over_speed_denominator = 1.0 / np.sqrt( \
     0.5 * (1.0 + np.sqrt(1.0 - 4.0 * aux_speed_b0**2 * aux_speed_b1**2)))
    alfven_speed = aux_speed_b0 * one_over_speed_denominator
    phase = wavenumber * (np.dot(unit_B0, x) - alfven_speed * t)
    fluid_velocity = -aux_speed_b1 * one_over_speed_denominator
    return fluid_velocity * (np.cos(phase) * unit_B1 - np.sin(phase) * unit_E)


def alfven_specific_internal_energy(x, t, wavenumber, pressure,
                                    rest_mass_density, adiabatic_index,
                                    bkgd_magnetic_field, wave_magnetic_field):
    return pressure / (rest_mass_density * (adiabatic_index - 1.0))


def alfven_pressure(x, t, wavenumber, pressure, rest_mass_density,
                    adiabatic_index, bkgd_magnetic_field, wave_magnetic_field):
    return pressure


def alfven_specific_enthalpy(x, t, wavenumber, pressure, rest_mass_density,
                             adiabatic_index, bkgd_magnetic_field,
                             wave_magnetic_field):
    return 1.0 + adiabatic_index * alfven_specific_internal_energy(
        x, t, wavenumber, pressure, rest_mass_density, adiabatic_index,
        bkgd_magnetic_field, wave_magnetic_field)


def alfven_lorentz_factor(x, t, wavenumber, pressure, rest_mass_density,
                          adiabatic_index, bkgd_magnetic_field,
                          wave_magnetic_field):
    return 1.0 / np.sqrt(1.0 - np.linalg.norm(
        alfven_spatial_velocity(x, t, wavenumber, pressure, rest_mass_density,
                                adiabatic_index, bkgd_magnetic_field,
                                wave_magnetic_field))**2)


def alfven_magnetic_field(x, t, wavenumber, pressure, rest_mass_density,
                          adiabatic_index, bkgd_magnetic_field,
                          wave_magnetic_field):
    magnitude_B0 = np.linalg.norm(bkgd_magnetic_field)
    magnitude_B1 = np.linalg.norm(wave_magnetic_field)
    unit_B0 = np.array(bkgd_magnetic_field) / magnitude_B0
    unit_B1 = np.array(wave_magnetic_field) / magnitude_B1
    unit_E = np.cross(unit_B1, unit_B0)
    rho_zero_times_h = \
     (rest_mass_density + pressure *
      (adiabatic_index)/(adiabatic_index - 1.0))
    aux_speed_b0 = magnitude_B0 / \
     np.sqrt(rho_zero_times_h + magnitude_B0 ** 2 + \
     magnitude_B1 ** 2)
    aux_speed_b1 = magnitude_B1 * aux_speed_b0 / magnitude_B0
    one_over_speed_denominator = 1.0 / np.sqrt( \
     0.5 * (1.0 + np.sqrt(1.0 - 4.0 * aux_speed_b0**2 * aux_speed_b1 **2)))
    alfven_speed = aux_speed_b0 * one_over_speed_denominator
    phase = wavenumber * (np.dot(unit_B0, x) - alfven_speed * t)
    return np.array(bkgd_magnetic_field) + \
      magnitude_B1 * (np.cos(phase) * unit_B1 - np.sin(phase) * unit_E)


def alfven_divergence_cleaning_field(x, t, wavenumber, pressure,
                                     rest_mass_density, adiabatic_index,
                                     bkgd_magnetic_field, wave_magnetic_field):
    return 0.0


# End functions for testing AlfvenWave.cpp


# Functions for testing BondiMichelMHD.cpp


def bondi_michel_sonic_fluid_speed_squared(r_c, mass):
    return 0.5 * mass / r_c


def bondi_michel_sonic_sound_speed_squared(u_c_2):
    return u_c_2 / (1.0 - 3.0 * u_c_2)


def bondi_michel_sonic_newtonian_sound_speed_squared(g, c_s_c_2):
    return (g - 1.0) * c_s_c_2 / (g - 1.0 - c_s_c_2)


def bondi_michel_adiabatic_constant(n_s_c_2, g, rho_c):
    return n_s_c_2 * pow(rho_c, 1.0 - g) / g


def bondi_michel_mass_accretion_rate_over_four_pi(r_c, rho_c, u_c_2):
    return r_c**2.0 * rho_c * np.sqrt(u_c_2)


def bondi_michel_bernoulli_constant_squared_minus_one(n_s_c_2, u_c_2, g):
    return -3.0 * u_c_2 * (1.0 + n_s_c_2/(g - 1.0))**2 + \
      (n_s_c_2 / (g - 1.0) * (2.0 + n_s_c_2 / (g - 1.0)))


def bondi_michel_bernoulli_equation_lhs_squared_minus_one(rho, r, g, k,
                                                          m_dot_over_four_pi,
                                                          mass):
    u = m_dot_over_four_pi / (r**2 * rho)
    g_minus_one = g - 1.0
    #polytropic_index times newtonian_sound_speed_squared:
    ncns2 = k * g * pow(rho, g_minus_one) / g_minus_one
    #specific enthalpy squared minus one:
    h2_minus_one = ncns2 * (2.0 + ncns2)
    h2 = h2_minus_one + 1.0
    return h2_minus_one + h2 * (-2.0 * mass / r + u**2)

def bondi_michel_bernoulli_root_function(rho, r, g, k, m_dot_over_four_pi,
                                         n_s_c_2, u_c_2, mass):
    return bondi_michel_bernoulli_equation_lhs_squared_minus_one(
        rho, r, g, k, m_dot_over_four_pi, mass) \
        - bondi_michel_bernoulli_constant_squared_minus_one(n_s_c_2, u_c_2, g)

def bondi_michel_sound_speed_at_infinity_squared(g, c_s_c_2):
    g_minus_one = g - 1.0
    return g_minus_one + (c_s_c_2 - g_minus_one) * np.sqrt(1.0 + 3.0 * c_s_c_2)


def bondi_michel_rest_mass_density_at_infinity(g, rho_c, c_s_inf_2, c_s_c_2):
    return rho_c * pow(c_s_inf_2 / (c_s_c_2 * np.sqrt(1.0 + 3.0 * c_s_c_2)),
                       1.0 / (g - 1.0))

def bondi_michel_rest_mass_density_at_horizon(g, rho_c, u_c_2):
    return 0.0625 * rho_c * pow(u_c_2, -1.5)


def bondi_michel_one_over_lapse(mass, radius):
    return np.sqrt(1.0 + 2.0 * mass / radius)


def bondi_michel_shift(mass, radius):
    return 2.0 * mass / (radius + 2.0 * mass)


def bondi_michel_fluid_four_velocity_u_t(mass, radius, abs_u_r):
    if radius == 2.0 * mass:
       return abs_u_r + 0.5 / abs_u_r
    return (-2.0 * mass * abs_u_r + radius * \
           np.sqrt(abs_u_r**2 + 1.0 - 2.0 * mass / radius)) / \
           (radius - 2.0 * mass)


def bondi_michel_intermediate_variables(mass, sonic_radius, sonic_density,
                                        adiabatic_exponent):
    u_c_2 = bondi_michel_sonic_fluid_speed_squared(sonic_radius, mass)
    c_s_c_2 = bondi_michel_sonic_sound_speed_squared(u_c_2)
    n_s_c_2 = bondi_michel_sonic_newtonian_sound_speed_squared(
        adiabatic_exponent, c_s_c_2)
    k = bondi_michel_adiabatic_constant(n_s_c_2, adiabatic_exponent,
                                        sonic_density)
    m_dot_over_four_pi = bondi_michel_mass_accretion_rate_over_four_pi(
        sonic_radius, sonic_density, u_c_2)
    h_inf_2 = bondi_michel_bernoulli_constant_squared_minus_one(
        n_s_c_2, u_c_2, adiabatic_exponent) + 1.0
    c_s_inf_2 = bondi_michel_sound_speed_at_infinity_squared(adiabatic_exponent,
                                                             c_s_c_2)
    rho_inf = bondi_michel_rest_mass_density_at_infinity(adiabatic_exponent,
                                                         sonic_density,
                                                         c_s_inf_2, c_s_c_2)
    rho_horizon = bondi_michel_rest_mass_density_at_horizon(adiabatic_exponent,
                                                            sonic_density,
                                                            u_c_2)
    return {"sonic_fluid_speed_squared": u_c_2, \
            "sonic_sound_speed_squared": c_s_c_2, \
            "sonic_newtonian_sound_speed_squared": n_s_c_2, \
            "adiabatic_constant": k,\
            "mass_accretion_rate_over_four_pi": m_dot_over_four_pi, \
            "bernoulli_constant_squared": h_inf_2, \
            "sound_speed_at_infinity_squared": c_s_inf_2, \
            "rest_mass_density_at_infinity": rho_inf, \
            "rest_mass_density_at_horizon": rho_horizon}


def bondi_michel_rest_mass_density(x, mass, sonic_radius, sonic_density,
                                   adiabatic_exponent, magnetic_field):
    variables = bondi_michel_intermediate_variables(mass, sonic_radius,
                                                    sonic_density,
                                                    adiabatic_exponent)
    radius = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    if radius < sonic_radius:
      upper_bound = variables["mass_accretion_rate_over_four_pi"] * \
        np.sqrt(2.0 / (mass * radius**3))
      lower_bound = variables["rest_mass_density_at_infinity"]
    if radius >= sonic_radius:
      upper_bound = sonic_density
      lower_bound = variables["mass_accretion_rate_over_four_pi"] * \
        np.sqrt(2.0 / (mass * radius**3))
    return opt.brentq( \
      bondi_michel_bernoulli_root_function, \
      lower_bound, upper_bound, xtol = 1.e-15, rtol = 1.e-15, args = ( \
        radius,
        adiabatic_exponent, \
        variables["adiabatic_constant"], \
        variables["mass_accretion_rate_over_four_pi"], \
        variables["sonic_newtonian_sound_speed_squared"], \
        variables["sonic_fluid_speed_squared"], \
        mass))

def bondi_michel_lorentz_factor(x, mass, sonic_radius, sonic_density,
                                adiabatic_exponent, magnetic_field):
    variables = bondi_michel_intermediate_variables(mass, sonic_radius,
                                                    sonic_density,
                                                    adiabatic_exponent)
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    fluid_speed_u = variables["mass_accretion_rate_over_four_pi"] / \
        (r**2 * bondi_michel_rest_mass_density(x, mass, sonic_radius,
                                               sonic_density,
                                               adiabatic_exponent,
                                               magnetic_field))
    return bondi_michel_fluid_four_velocity_u_t(mass, r, fluid_speed_u) / \
        bondi_michel_one_over_lapse(mass, r)


def bondi_michel_spatial_velocity(x, mass, sonic_radius, sonic_density,
                                  adiabatic_exponent, magnetic_field):
    variables = bondi_michel_intermediate_variables(mass, sonic_radius,
                                                    sonic_density,
                                                    adiabatic_exponent)
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    fluid_speed_u = variables["mass_accretion_rate_over_four_pi"] / \
        (r**2 * bondi_michel_rest_mass_density(x, mass, sonic_radius,
                                               sonic_density,
                                               adiabatic_exponent,
                                               magnetic_field))
    #eulerian_radial_velocity_over_radius
    e = bondi_michel_one_over_lapse(mass, r) * \
        (-fluid_speed_u / bondi_michel_fluid_four_velocity_u_t(mass, r,
                                                               fluid_speed_u) +
         bondi_michel_shift(mass, r)) / r
    return np.array([e * x[0], e * x[1], e * x[2]])


def bondi_michel_specific_internal_energy(x, mass, sonic_radius, sonic_density,
                                          adiabatic_exponent, magnetic_field):
    rest_mass_density = bondi_michel_rest_mass_density(x, mass, sonic_radius,
                                                       sonic_density,
                                                       adiabatic_exponent,
                                                       magnetic_field)
    pressure = bondi_michel_pressure(x, mass, sonic_radius, sonic_density,
                                     adiabatic_exponent, magnetic_field)
    return pressure / (rest_mass_density * (adiabatic_exponent - 1.0))


def bondi_michel_specific_enthalpy(x, mass, sonic_radius, sonic_density,
                                   adiabatic_exponent, magnetic_field):
    return 1.0 + adiabatic_exponent * bondi_michel_specific_internal_energy(
        x, mass, sonic_radius, sonic_density, adiabatic_exponent,
        magnetic_field)


def bondi_michel_pressure(x, mass, sonic_radius, sonic_density,
                          adiabatic_exponent, magnetic_field):
    rest_mass_density = bondi_michel_rest_mass_density(x, mass, sonic_radius,
                                                       sonic_density,
                                                       adiabatic_exponent,
                                                       magnetic_field)
    variables = bondi_michel_intermediate_variables(mass, sonic_radius,
                                                    sonic_density,
                                                    adiabatic_exponent)
    return variables["adiabatic_constant"] * \
        pow(rest_mass_density, adiabatic_exponent)


def bondi_michel_divergence_cleaning_field(x, mass, sonic_radius, sonic_density,
                                           adiabatic_exponent, magnetic_field):
    return 0.0


def bondi_michel_magnetic_field(x, mass, sonic_radius, sonic_density,
                                adiabatic_exponent, magnetic_field):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    m = magnetic_field / (r**3 * np.sqrt(1.0 + 2.0/r))
    return np.array([m * x[0], m * x[1], m * x[2]])


# End functions for testing BondiMichelMHD.cpp
