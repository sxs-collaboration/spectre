# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Note: Values for p_* and u_* correspond to those
# values obtained for the initial data of the Sod tube test.


def sound_speed(adiabatic_index, mass_density, pressure):
    return np.sqrt(adiabatic_index * pressure / mass_density)


def fan_density(x_shifted, t, adiabatic_index, mass_density, velocity,
                pressure, direction):
    s = (x_shifted / t) if t > 0.0 else 0.0
    return (mass_density * np.power(
        (2.0 - direction * (adiabatic_index - 1.0) *
         (velocity - s) / sound_speed(adiabatic_index, mass_density, pressure))
        / (adiabatic_index + 1.0), 2.0 / (adiabatic_index - 1.0)))


def fan_velocity(x_shifted, t, adiabatic_index, mass_density, velocity,
                 pressure, direction):
    s = (x_shifted / t) if t > 0.0 else 0.0
    return (
        2.0 *
        (0.5 * (adiabatic_index - 1.0) * velocity + s -
         direction * sound_speed(adiabatic_index, mass_density, pressure)) /
        (adiabatic_index + 1.0))


def fan_pressure(x_shifted, t, adiabatic_index, mass_density, velocity,
                 pressure, direction):
    s = (x_shifted / t) if t > 0.0 else 0.0
    return (pressure * np.power(
        (2.0 - direction * (adiabatic_index - 1.0) *
         (velocity - s) / sound_speed(adiabatic_index, mass_density, pressure))
        / (adiabatic_index + 1.0), 2.0 * adiabatic_index /
        (adiabatic_index - 1.0)))


def rarefaction(x_shifted, t, quantity, quantity_star, adiabatic_index,
                velocity_star, pressure_star, fan, mass_density, velocity,
                pressure, direction):
    pressure_ratio = pressure_star / pressure
    sound_speed_ = sound_speed(adiabatic_index, mass_density, pressure)
    sound_speed_star = sound_speed_ * np.power(
        pressure_ratio, 0.5 * (adiabatic_index - 1.0) / adiabatic_index)
    head_speed = velocity + direction * sound_speed_
    tail_speed = velocity_star + direction * sound_speed_star
    return (quantity_star if direction *
            (x_shifted - tail_speed * t) < 0.0 else
            (fan(x_shifted, t, adiabatic_index, mass_density, velocity,
                 pressure, direction) if
             (direction * (x_shifted - tail_speed * t) >= 0.0 and direction *
              (x_shifted - head_speed * t) < 0.0) else quantity))


def shock(x_shifted, t, quantity, quantity_star, adiabatic_index,
          pressure_star, mass_density, velocity, pressure, direction):
    shock_speed = (
        velocity +
        direction * sound_speed(adiabatic_index, mass_density, pressure) *
        np.sqrt(0.5 * ((adiabatic_index + 1.0) *
                       (pressure_star / pressure) + adiabatic_index - 1.0) /
                adiabatic_index))
    return (quantity_star if direction *
            (x_shifted - shock_speed * t) < 0.0 else quantity)


def mass_density(x, t, adiabatic_index, initial_pos, l_mass_density,
                 l_velocity, l_pressure, r_mass_density, r_velocity,
                 r_pressure):
    velocity_star = 0.9274526526
    pressure_star = 0.3031302631

    # density in star region for shock
    gamma_mm_over_gamma_pp = (adiabatic_index - 1.0) / (adiabatic_index + 1.0)
    r_pressure_ratio = pressure_star / r_pressure
    r_density_star = (r_mass_density *
                      (r_pressure_ratio + gamma_mm_over_gamma_pp) /
                      (r_pressure_ratio * gamma_mm_over_gamma_pp + 1.0))

    # density in star region for rarefaction
    l_density_star = l_mass_density * np.power(pressure_star / l_pressure,
                                               1.0 / adiabatic_index)

    x_shifted = x[0] - initial_pos
    return (rarefaction(x_shifted, t, l_mass_density, l_density_star,
                        adiabatic_index, velocity_star, pressure_star,
                        fan_density, l_mass_density, l_velocity[0], l_pressure,
                        -1.0)
            if x_shifted < velocity_star * t else shock(
                x_shifted, t, r_mass_density, r_density_star, adiabatic_index,
                pressure_star, r_mass_density, r_velocity[0], r_pressure, 1.0))


def velocity(x, t, adiabatic_index, initial_pos, l_mass_density, l_velocity,
             l_pressure, r_mass_density, r_velocity, r_pressure):
    velocity_star = 0.9274526526
    pressure_star = 0.3031302631

    velocity = np.zeros(x.size)
    x_shifted = x[0] - initial_pos
    velocity[0] = (rarefaction(
        x_shifted, t, l_velocity[0], velocity_star, adiabatic_index,
        velocity_star, pressure_star, fan_velocity, l_mass_density,
        l_velocity[0], l_pressure, -1.0) if x_shifted < velocity_star * t else
                   shock(x_shifted, t, r_velocity[0], velocity_star,
                         adiabatic_index, pressure_star, r_mass_density,
                         r_velocity[0], r_pressure, 1.0))
    if (x.size > 1):
        velocity[1] = (l_velocity[1]
                       if x_shifted < velocity_star * t else r_velocity[1])
    if (x.size > 2):
        velocity[2] = (l_velocity[2]
                       if x_shifted < velocity_star * t else r_velocity[2])
    return velocity


def pressure(x, t, adiabatic_index, initial_pos, l_mass_density, l_velocity,
             l_pressure, r_mass_density, r_velocity, r_pressure):
    velocity_star = 0.9274526526
    pressure_star = 0.3031302631

    x_shifted = x[0] - initial_pos
    return (rarefaction(x_shifted, t, l_pressure, pressure_star,
                        adiabatic_index, velocity_star, pressure_star,
                        fan_pressure, l_mass_density, l_velocity[0],
                        l_pressure, -1.0)
            if x_shifted < velocity_star * t else shock(
                x_shifted, t, r_pressure, pressure_star, adiabatic_index,
                pressure_star, r_mass_density, r_velocity[0], r_pressure, 1.0))


def specific_internal_energy(x, t, adiabatic_index, initial_pos,
                             l_mass_density, l_velocity, l_pressure,
                             r_mass_density, r_velocity, r_pressure):
    return (pressure(
        x, t, adiabatic_index, initial_pos, l_mass_density, l_velocity,
        l_pressure, r_mass_density, r_velocity, r_pressure) / mass_density(
            x, t, adiabatic_index, initial_pos, l_mass_density, l_velocity,
            l_pressure, r_mass_density, r_velocity, r_pressure) /
            (adiabatic_index - 1.0))
