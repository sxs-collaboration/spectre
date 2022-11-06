# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def characteristic_speeds(gamma_1, lapse, shift, unit_normal_one_form):
    shift_dot_normal = np.dot(shift, unit_normal_one_form)
    return [
        -(1.0 + gamma_1) * shift_dot_normal, -shift_dot_normal,
        -shift_dot_normal + lapse, -shift_dot_normal - lapse
    ]


def error(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, gamma_1, lapse, shift):
    speeds = characteristic_speeds(gamma_1, lapse, shift,
                                   outward_directed_normal_covector)
    for i in range(4):
        if face_mesh_velocity is not None:
            speeds[i] -= np.dot(outward_directed_normal_covector,
                                face_mesh_velocity)
            speeds[0] -= np.dot(outward_directed_normal_covector,
                                face_mesh_velocity) * gamma_1
        if speeds[i] < 0.0:
            return ("DemandOutgoingCharSpeeds boundary condition violated")
    return None

    pass
