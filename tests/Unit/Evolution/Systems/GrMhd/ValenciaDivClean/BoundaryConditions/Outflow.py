# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def characteristic_speeds(lapse, shift, outward_directed_normal_covector):
    return [
        lapse - np.dot(outward_directed_normal_covector, shift),
        -lapse - np.dot(outward_directed_normal_covector, shift)
    ]


def error(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, shift, lapse):
    speeds = characteristic_speeds(lapse, shift,
                                   outward_directed_normal_covector)
    for i in range(2):
        if face_mesh_velocity is not None:
            speeds[i] -= np.dot(outward_directed_normal_covector,
                                face_mesh_velocity)
        if speeds[i] < 0.0:
            return ("Outflow boundary condition violated. .*")
    return None
