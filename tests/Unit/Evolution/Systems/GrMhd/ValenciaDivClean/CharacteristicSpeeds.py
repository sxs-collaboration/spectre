# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def CharacteristicSpeeds(
    lapse,
    shift,
    spatial_velocity,
    spatial_velocity_sqrd,
    sound_speed_sqrd,
    alfven_speed_sqrd,
    normal_oneform,
):
    normal_velocity = np.dot(spatial_velocity, normal_oneform)
    normal_shift = np.dot(shift, normal_oneform)
    sound_speed_sqrd += alfven_speed_sqrd * (1.0 - sound_speed_sqrd)
    prefactor = lapse / (1.0 - spatial_velocity_sqrd * sound_speed_sqrd)
    first_term = prefactor * normal_velocity * (1.0 - sound_speed_sqrd)
    second_term = (
        prefactor
        * np.sqrt(sound_speed_sqrd)
        * np.sqrt(
            (1.0 - spatial_velocity_sqrd)
            * (
                1.0
                - spatial_velocity_sqrd * sound_speed_sqrd
                - normal_velocity * normal_velocity * (1.0 - sound_speed_sqrd)
            )
        )
    )
    result = [-lapse - normal_shift]
    result.append(first_term - second_term - normal_shift)
    for i in range(0, spatial_velocity.size + 2):
        result.append(lapse * normal_velocity - normal_shift)
    result.append(first_term + second_term - normal_shift)
    result.append(lapse - normal_shift)
    return result


def characteristic_speeds_hydro(
    spatial_velocity,
    spatial_velocity_squared,
    sound_speed_squared,
    lorentz_factor,
    normal_oneform,
):
    normal_velocity = np.dot(spatial_velocity, normal_oneform)
    denom = 1.0 - spatial_velocity_squared * sound_speed_squared
    under = (
        1.0
        - spatial_velocity_squared * sound_speed_squared
        - normal_velocity * normal_velocity * (1.0 - sound_speed_squared)
    )

    d = lorentz_factor * np.sqrt(under)
    factor = (1.0 - sound_speed_squared) * normal_velocity
    cs = np.sqrt(sound_speed_squared)

    y_plus = (factor + cs * d / (lorentz_factor * lorentz_factor)) / denom
    y_minus = (factor - cs * d / (lorentz_factor * lorentz_factor)) / denom

    return [
        normal_velocity,
        float(y_plus),
        float(y_minus),
    ]
