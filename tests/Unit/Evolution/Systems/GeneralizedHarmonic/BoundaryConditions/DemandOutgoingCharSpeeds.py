# Distributed under the MIT License.
# See LICENSE.txt for details.

import Evolution.Systems.GeneralizedHarmonic.Characteristics as ght
import numpy as np


def characteristic_speeds(gamma_1, lapse, shift, unit_normal_one_form):
    return [
        ght.char_speed_upsi(gamma_1, lapse, shift, unit_normal_one_form),
        ght.char_speed_uzero(gamma_1, lapse, shift, unit_normal_one_form),
        ght.char_speed_uplus(gamma_1, lapse, shift, unit_normal_one_form),
        ght.char_speed_uminus(gamma_1, lapse, shift, unit_normal_one_form),
    ]


def error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    gamma_1,
    lapse,
    shift,
):
    if face_mesh_velocity is not None:
        char_speeds = [
            ght.char_speed_upsi_moving_mesh(
                gamma_1,
                lapse,
                shift,
                outward_directed_normal_covector,
                face_mesh_velocity,
            ),
            ght.char_speed_uzero_moving_mesh(
                gamma_1,
                lapse,
                shift,
                outward_directed_normal_covector,
                face_mesh_velocity,
            ),
            ght.char_speed_uplus_moving_mesh(
                gamma_1,
                lapse,
                shift,
                outward_directed_normal_covector,
                face_mesh_velocity,
            ),
            ght.char_speed_uminus_moving_mesh(
                gamma_1,
                lapse,
                shift,
                outward_directed_normal_covector,
                face_mesh_velocity,
            ),
        ]
    else:
        char_speeds = [
            ght.char_speed_upsi(
                gamma_1, lapse, shift, outward_directed_normal_covector
            ),
            ght.char_speed_uzero(
                gamma_1, lapse, shift, outward_directed_normal_covector
            ),
            ght.char_speed_uplus(
                gamma_1, lapse, shift, outward_directed_normal_covector
            ),
            ght.char_speed_uminus(
                gamma_1, lapse, shift, outward_directed_normal_covector
            ),
        ]
    for i in range(4):
        if char_speeds[i] < 0.0:
            return "DemandOutgoingCharSpeeds boundary condition violated"
    return None

    pass
