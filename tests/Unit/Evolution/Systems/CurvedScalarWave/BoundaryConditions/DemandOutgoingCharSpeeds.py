# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Evolution.Systems.CurvedScalarWave.Characteristics import (
    char_speed_vpsi, char_speed_vzero, char_speed_vplus, char_speed_vminus)


def error(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, gamma_1, lapse, shift):
    speeds = np.zeros(4)
    speeds[0] = char_speed_vpsi(gamma_1, lapse, shift,
                                outward_directed_normal_covector)
    speeds[1] = char_speed_vzero(gamma_1, lapse, shift,
                                 outward_directed_normal_covector)
    speeds[2] = char_speed_vplus(gamma_1, lapse, shift,
                                 outward_directed_normal_covector)
    speeds[3] = char_speed_vminus(gamma_1, lapse, shift,
                                  outward_directed_normal_covector)
    for speed in speeds:
        if face_mesh_velocity is not None:
            speed -= np.dot(outward_directed_normal_covector,
                            face_mesh_velocity)
        if speed < 0.0:
            return ("Detected negative characteristic speed *")
    return None
