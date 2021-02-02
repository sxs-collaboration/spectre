# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error_sinusoid(face_mesh_velocity, outward_directed_normal_covector,
                   coords, time, ignored_argument_for_analytic_data):
    return None


def u_sinusoid(face_mesh_velocity, outward_directed_normal_covector, coords,
               time, ignored_argument_for_analytic_data):
    return np.sin(coords[0])


def flux_sinusoid(face_mesh_velocity, outward_directed_normal_covector, coords,
                  time, ignored_argument_for_analytic_data):
    u = np.sin(coords[0])
    return np.asarray([0.5 * u**2])


def error_step(face_mesh_velocity, outward_directed_normal_covector, coords,
               time, ignored_argument_for_analytic_data):
    return None


def u_step(face_mesh_velocity, outward_directed_normal_covector, coords, time,
           analytic_soln_params):
    current_shock_position = analytic_soln_params[2] + 0.5 * (
        analytic_soln_params[0] + analytic_soln_params[1]) * time
    if coords[0] - current_shock_position < 0.0:
        return analytic_soln_params[0]
    else:
        return analytic_soln_params[1]


def flux_step(face_mesh_velocity, outward_directed_normal_covector, coords,
              time, analytic_soln_params):
    return np.asarray([
        0.5 * u_step(face_mesh_velocity, outward_directed_normal_covector,
                     coords, time, analytic_soln_params)**2
    ])
