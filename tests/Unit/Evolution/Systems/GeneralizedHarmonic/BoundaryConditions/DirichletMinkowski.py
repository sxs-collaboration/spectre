# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import PointwiseFunctions.GeneralRelativity.ComputeSpacetimeQuantities as gr


def error(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, interior_gamma1, interior_gamma2):
    return None


def lapse(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, interior_gamma1, interior_gamma2):
    return 1.0


def shift(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, interior_gamma1, interior_gamma2):
    return np.zeros_like(outward_directed_normal_covector)


def spacetime_metric(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, interior_gamma1,
                     interior_gamma2):
    return gr.spacetime_metric(
        -1.0, np.zeros_like(outward_directed_normal_covector),
        np.diag(np.ones(len(outward_directed_normal_covector))))


def phi(face_mesh_velocity, outward_directed_normal_covector,
        outward_directed_normal_vector, interior_gamma1, interior_gamma2):
    return np.zeros((len(outward_directed_normal_covector),
                     len(outward_directed_normal_covector) + 1,
                     len(outward_directed_normal_covector) + 1))


def pi(face_mesh_velocity, outward_directed_normal_covector,
       outward_directed_normal_vector, interior_gamma1, interior_gamma2):
    return np.zeros((len(outward_directed_normal_covector) + 1,
                     len(outward_directed_normal_covector) + 1))


def constraint_gamma1(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, interior_gamma1,
                      interior_gamma2):
    assert interior_gamma1 >= 0.0
    return interior_gamma1


def constraint_gamma2(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, interior_gamma1,
                      interior_gamma2):
    assert interior_gamma2 >= 0.0
    return interior_gamma2
