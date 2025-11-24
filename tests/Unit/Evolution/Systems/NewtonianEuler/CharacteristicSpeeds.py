# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def CharacteristicSpeeds(velocity, sound_speed, normal):
    normal_velocity = np.dot(velocity, normal)
    result = [normal_velocity - sound_speed]
    for i in range(0, velocity.size):
        result.append(normal_velocity)
    result.append(normal_velocity + sound_speed)
    return result
