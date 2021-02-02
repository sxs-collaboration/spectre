# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(face_mesh_velocity, outward_directed_normal_covector, u):
    speed = outward_directed_normal_covector[0] * u
    if not face_mesh_velocity is None:
        speed -= face_mesh_velocity[0] * outward_directed_normal_covector[0]

    if speed < 0.0:
        return "Outflow boundary condition violated with speed U ingoing:.*"
    return None
