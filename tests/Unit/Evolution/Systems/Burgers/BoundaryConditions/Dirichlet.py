# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(face_mesh_velocity, outward_directed_normal_covector):
    return None


def flux_1(face_mesh_velocity, outward_directed_normal_covector):
    return np.asarray([0.5])


def u_1(face_mesh_velocity, outward_directed_normal_covector):
    return 1.0


def flux_m1(face_mesh_velocity, outward_directed_normal_covector):
    return np.asarray([0.5])


def u_m1(face_mesh_velocity, outward_directed_normal_covector):
    return -1.0
