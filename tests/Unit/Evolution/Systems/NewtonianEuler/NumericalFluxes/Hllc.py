# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from Evolution.Systems.NewtonianEuler.TestFunctions import characteristic_speeds


def sound_speed(mass_density, pressure):
    return np.sqrt(1.3333333333333333 * pressure / mass_density)


def signal_speed_star(mass_density_int, pressure_int, n_dot_f_mass_density_int,
                      normal_velocity_int, signal_speed_int,
                      mass_density_ext, pressure_ext, n_dot_f_mass_density_ext,
                      normal_velocity_ext, signal_speed_ext):
    return ((pressure_int + n_dot_f_mass_density_int *
             (normal_velocity_int - signal_speed_int) -
             pressure_ext - n_dot_f_mass_density_ext *
             (normal_velocity_ext - signal_speed_ext)) /
            (n_dot_f_mass_density_int - signal_speed_int * mass_density_int -
             n_dot_f_mass_density_ext + signal_speed_ext * mass_density_ext))


def pressure_star(mass_density, pressure, normal_velocity, signal_speed,
                  signal_speed_star):
    return (pressure + mass_density * (normal_velocity - signal_speed) *
            (normal_velocity - signal_speed_star))


def flux_star(n_dot_f, u, signal_speed, d, signal_speed_star, p_star):
    return (signal_speed_star * (n_dot_f - signal_speed * u) -
            signal_speed * p_star * d) / (signal_speed_star - signal_speed)


def apply_flux(n_dot_f_int, u_int, vector_d_int, mass_density_int,
               momentum_density_int, velocity_int, pressure_int,
               sound_speed_int, n_dot_f_mass_density_int, normal_int,
               n_dot_f_ext, u_ext, vector_d_ext, mass_density_ext,
               momentum_density_ext, velocity_ext, pressure_ext,
               sound_speed_ext, n_dot_f_mass_density_ext, normal_ext):
    normal_velocity_int = np.dot(velocity_int, normal_int)
    normal_velocity_ext = np.dot(velocity_ext, normal_ext)
    all_char_speeds = (characteristic_speeds(velocity_int, sound_speed_int,
                                             normal_int) +
                       characteristic_speeds(velocity_ext, sound_speed_ext,
                                             normal_ext))
    c_min = min(all_char_speeds)
    c_max = max(all_char_speeds)
    c_star = signal_speed_star(mass_density_int, pressure_int,
                               n_dot_f_mass_density_int, normal_velocity_int,
                               c_min, mass_density_ext, pressure_ext,
                               n_dot_f_mass_density_ext, normal_velocity_ext,
                               c_max)
    if (c_min >= 0.0):
        return n_dot_f_int
    elif (c_max <= 0.0):
        return n_dot_f_ext
    elif (c_min < 0.0 and c_star >= 0.0):
        return flux_star(n_dot_f_int, u_int, c_min, vector_d_int, c_star,
                         pressure_star(mass_density_int, pressure_int,
                                       normal_velocity_int, c_min, c_star))
    elif(c_star < 0.0 and c_max > 0.0):
        return flux_star(n_dot_f_ext, u_ext, c_max, vector_d_ext, c_star,
                         pressure_star(mass_density_ext, pressure_ext,
                                       normal_velocity_ext, c_max, c_star))
    else:
        raise Error("Signal speeds not valid.")


def n_dot_num_f_mass_density(mass_density_int, momentum_density_int,
                             energy_density_int, pressure_int, mass_density_ext,
                             momentum_density_ext, energy_density_ext,
                             pressure_ext, interface_normal):
    velocity_int = momentum_density_int / mass_density_int
    normal_velocity_int = np.dot(velocity_int, interface_normal)
    velocity_ext = momentum_density_ext / mass_density_ext
    normal_velocity_ext = np.dot(velocity_ext, interface_normal)
    return apply_flux(mass_density_int * normal_velocity_int, mass_density_int,
                      0.0, mass_density_int, momentum_density_int, velocity_int,
                      pressure_int, sound_speed(mass_density_int, pressure_int),
                      mass_density_int * normal_velocity_int, interface_normal,
                      mass_density_ext * normal_velocity_ext, mass_density_ext,
                      0.0, mass_density_ext, momentum_density_ext, velocity_ext,
                      pressure_ext, sound_speed(mass_density_ext, pressure_ext),
                      mass_density_ext * normal_velocity_ext, interface_normal)


def n_dot_num_f_momentum_density(mass_density_int, momentum_density_int,
                                 energy_density_int, pressure_int,
                                 mass_density_ext, momentum_density_ext,
                                 energy_density_ext, pressure_ext,
                                 interface_normal):
    velocity_int = momentum_density_int / mass_density_int
    normal_velocity_int = np.dot(velocity_int, interface_normal)
    velocity_ext = momentum_density_ext / mass_density_ext
    normal_velocity_ext = np.dot(velocity_ext, interface_normal)
    return apply_flux(momentum_density_int * normal_velocity_int +
                      pressure_int * interface_normal, momentum_density_int,
                      interface_normal, mass_density_int, momentum_density_int,
                      velocity_int, pressure_int,
                      sound_speed(mass_density_int, pressure_int),
                      mass_density_int * normal_velocity_int, interface_normal,
                      momentum_density_ext * normal_velocity_ext +
                      pressure_ext * interface_normal, momentum_density_ext,
                      interface_normal, mass_density_ext, momentum_density_ext,
                      velocity_ext, pressure_ext,
                      sound_speed(mass_density_ext, pressure_ext),
                      mass_density_ext * normal_velocity_ext, interface_normal)


def n_dot_num_f_energy_density(mass_density_int, momentum_density_int,
                               energy_density_int, pressure_int,
                               mass_density_ext, momentum_density_ext,
                               energy_density_ext, pressure_ext,
                               interface_normal):
    velocity_int = momentum_density_int / mass_density_int
    sound_speed_int = sound_speed(mass_density_int, pressure_int)
    normal_velocity_int = np.dot(velocity_int, interface_normal)
    velocity_ext = momentum_density_ext / mass_density_ext
    sound_speed_ext = sound_speed(mass_density_ext, pressure_ext)
    normal_velocity_ext = np.dot(velocity_ext, interface_normal)
    all_char_speeds = (characteristic_speeds(velocity_int, sound_speed_int,
                                             interface_normal) +
                       characteristic_speeds(velocity_ext, sound_speed_ext,
                                             interface_normal))
    c_min = min(all_char_speeds)
    c_max = max(all_char_speeds)
    c_star = signal_speed_star(mass_density_int, pressure_int,
                               mass_density_int * normal_velocity_int,
                               normal_velocity_int, c_min,
                               mass_density_ext, pressure_ext,
                               mass_density_ext * normal_velocity_ext,
                               normal_velocity_ext, c_max)
    return apply_flux((energy_density_int + pressure_int) * normal_velocity_int,
                      energy_density_int, c_star, mass_density_int,
                      momentum_density_int, velocity_int, pressure_int,
                      sound_speed_int, mass_density_int * normal_velocity_int,
                      interface_normal,
                      (energy_density_ext + pressure_ext) * normal_velocity_ext,
                      energy_density_ext, c_star, mass_density_ext,
                      momentum_density_ext, velocity_ext, pressure_ext,
                      sound_speed_ext, mass_density_ext * normal_velocity_ext,
                      interface_normal)


