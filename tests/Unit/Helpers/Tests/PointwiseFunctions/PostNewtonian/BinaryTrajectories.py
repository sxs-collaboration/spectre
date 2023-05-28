# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

initial_separation = 15.366
initial_velocity = np.array([0.1, -0.2, 0.3])


def separation(time, init_sep, newtonian):
    correction = 0.0 if newtonian else 12.8 * time
    return (init_sep**4 - correction) ** 0.25


def orbital_frequency(time, init_sep, newtonian):
    return separation(time, init_sep, newtonian) ** -1.5


def angular_velocity(time, init_sep, newtonian):
    correction = (
        0.0 if newtonian else 4.8 * (init_sep**4 - 12.8 * time) ** -1.375
    )
    return orbital_frequency(time, init_sep, newtonian) + correction * time


def position_helper(time, init_sep, newtonian, sign, no_expansion):
    sep = (
        separation(time, init_sep, True)
        if no_expansion
        else separation(time, init_sep, newtonian)
    )
    freq = orbital_frequency(time, init_sep, newtonian)
    return [
        sign * 0.5 * sep * np.cos(freq * time) + initial_velocity[0] * time,
        sign * 0.5 * sep * np.sin(freq * time) + initial_velocity[1] * time,
        initial_velocity[2] * time,
    ]


def positions1(time, newtonian, no_expansion):
    newt = True if newtonian > 0.0 else False
    no_exp = True if no_expansion > 0.0 else False
    return position_helper(time, initial_separation, newt, +1.0, no_exp)


def positions2(time, newtonian, no_expansion):
    newt = True if newtonian > 0.0 else False
    no_exp = True if no_expansion > 0.0 else False
    return position_helper(time, initial_separation, newt, -1.0, no_exp)
