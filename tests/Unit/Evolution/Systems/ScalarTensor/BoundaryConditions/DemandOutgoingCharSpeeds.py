# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def characteristic_speeds(gamma_1, lapse, shift, unit_normal_one_form):
    shift_dot_normal = np.dot(shift, unit_normal_one_form)
    return [
        -(1.0 + gamma_1) * shift_dot_normal,
        -shift_dot_normal,
        -shift_dot_normal + lapse,
        -shift_dot_normal - lapse,
    ]


def error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    gh_gamma_1,
    lapse,
    shift,
    csw_gamma_1,
):
    gh_speeds = characteristic_speeds(
        gh_gamma_1, lapse, shift, outward_directed_normal_covector
    )
    for i in range(4):
        if face_mesh_velocity is not None:
            gh_speeds[i] -= np.dot(
                outward_directed_normal_covector, face_mesh_velocity
            )
            gh_speeds[0] -= (
                np.dot(outward_directed_normal_covector, face_mesh_velocity)
                * gh_gamma_1
            )
        if gh_speeds[i] < 0.0:
            return (
                "Detected negative characteristic speed at boundary with "
                "outgoing char speeds boundary conditions specified. The "
                "speed is "
            )

    # Char speeds are the same for GH and CSW.
    csw_speeds = characteristic_speeds(
        csw_gamma_1, lapse, shift, outward_directed_normal_covector
    )
    for i in range(4):
        if face_mesh_velocity is not None:
            csw_speeds[i] -= np.dot(
                outward_directed_normal_covector, face_mesh_velocity
            )
            csw_speeds[0] -= (
                np.dot(outward_directed_normal_covector, face_mesh_velocity)
                * gh_gamma_1
            )
        if csw_speeds[i] < 0.0:
            return (
                "Detected negative characteristic speed at boundary with "
                "outgoing char speeds boundary conditions specified. The "
                "speed is "
            )
    return None

    pass
