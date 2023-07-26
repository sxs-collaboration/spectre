# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    inverse_spatial_metric_interior,
    gamma1_interior,
    gamma2_interior,
    lapse_interior,
    shift_interior,
):
    return None


def psi_analytic_constant(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    inverse_spatial_metric_interior,
    gamma1_interior,
    gamma2_interior,
    lapse_interior,
    shift_interior,
):
    return 2.71


def pi_analytic_constant(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    inverse_spatial_metric_interior,
    gamma1_interior,
    gamma2_interior,
    lapse_interior,
    shift_interior,
):
    return 0.0


def phi_analytic_constant(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    inverse_spatial_metric_interior,
    gamma1_interior,
    gamma2_interior,
    lapse_interior,
    shift_interior,
):
    return 0.0 * shift_interior


def lapse_analytic_constant(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    inverse_spatial_metric_interior,
    gamma1_interior,
    gamma2_interior,
    lapse_interior,
    shift_interior,
):
    return lapse_interior


def shift_analytic_constant(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    inverse_spatial_metric_interior,
    gamma1_interior,
    gamma2_interior,
    lapse_interior,
    shift_interior,
):
    return shift_interior


def constraint_gamma1_analytic_constant(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    inverse_spatial_metric_interior,
    gamma1_interior,
    gamma2_interior,
    lapse_interior,
    shift_interior,
):
    return gamma1_interior


def constraint_gamma2_analytic_constant(
    face_mesh_velocity,
    normal_covector,
    normal_vector,
    inverse_spatial_metric_interior,
    gamma1_interior,
    gamma2_interior,
    lapse_interior,
    shift_interior,
):
    return gamma2_interior
