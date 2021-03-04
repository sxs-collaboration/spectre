# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import sqrt, abs


def make_metric_riemannian(inv_conformal_metric):
    for i in range(3):
        for j in range(i):
            inv_conformal_metric[i, j] *= 1.e-2
            inv_conformal_metric[j, i] *= 1.e-2
        inv_conformal_metric[i, i] = abs(inv_conformal_metric[i, i])


def make_spherical_face_normal(x, inv_conformal_metric):
    proper_radius = sqrt(np.einsum('ij,i,j', inv_conformal_metric, x, x))
    return -x / proper_radius


# This is h^ij nabla_i s_j (all quantities conformal) in Eq. 7.12 in
# Harald's thesis https://arxiv.org/abs/gr-qc/0510016
def projected_normal_gradient(conformal_unit_normal, x, inv_conformal_metric,
                              conformal_christoffel_second_kind):
    conformal_unit_normal_raised = np.einsum('ij,j', inv_conformal_metric,
                                             conformal_unit_normal)
    inv_conformal_surface_metric = inv_conformal_metric - np.einsum(
        'i,j->ij', conformal_unit_normal_raised, conformal_unit_normal_raised)
    # Assuming here that the surface is a coordinate-sphere
    euclidean_radius = np.linalg.norm(x)
    unnormalized_face_normal = x / euclidean_radius
    magnitude_of_face_normal = sqrt(
        np.einsum('ij,i,j', inv_conformal_metric, unnormalized_face_normal,
                  unnormalized_face_normal))  # r_curved / r_flat
    deriv_unnormalized_face_normal = (
        np.identity(3) / euclidean_radius -
        np.einsum('i,j->ij', x, x) / euclidean_radius**3)
    # The term with the derivative of the magnitude vanishes when projected on
    # the surface metric, so it's omitted here
    conformal_unit_normal_gradient = (
        deriv_unnormalized_face_normal / magnitude_of_face_normal - np.einsum(
            'i,ijk', conformal_unit_normal, conformal_christoffel_second_kind))
    return np.einsum('ij,ij', inv_conformal_surface_metric,
                     conformal_unit_normal_gradient)


# This function implements the apparent-horizon boundary condition, Eq. 7.12 in
# Harald's thesis https://arxiv.org/abs/gr-qc/0510016
def normal_dot_conformal_factor_gradient(
    conformal_factor, lapse_times_conformal_factor,
    n_dot_longitudinal_shift_excess, center, spin, x,
    extrinsic_curvature_trace, shift_background, longitudinal_shift_background,
    inv_conformal_metric, conformal_christoffel_second_kind):
    x -= center
    make_metric_riemannian(inv_conformal_metric)
    conformal_unit_normal = -make_spherical_face_normal(
        x, inv_conformal_metric)
    local_projected_normal_gradient = projected_normal_gradient(
        conformal_unit_normal, x, inv_conformal_metric,
        conformal_christoffel_second_kind)
    lapse = lapse_times_conformal_factor / conformal_factor
    n_dot_longitudinal_shift = n_dot_longitudinal_shift_excess + np.einsum(
        'i,ij', -conformal_unit_normal, longitudinal_shift_background)
    J = (2. / 3. * extrinsic_curvature_trace - 0.5 / lapse *
         np.einsum('i,i', -conformal_unit_normal, n_dot_longitudinal_shift))
    return (0.25 * conformal_factor *
            (local_projected_normal_gradient - conformal_factor**2 * J))


def normal_dot_conformal_factor_gradient_flat_cartesian(*args, **kwargs):
    return normal_dot_conformal_factor_gradient(
        *args,
        inv_conformal_metric=np.identity(3),
        conformal_christoffel_second_kind=np.zeros((3, 3, 3)),
        **kwargs)


def normal_dot_conformal_factor_gradient_correction(
    conformal_factor_correction, lapse_times_conformal_factor_correction,
    n_dot_longitudinal_shift_excess_correction, center, spin, x,
    extrinsic_curvature_trace, longitudinal_shift_background, conformal_factor,
    lapse_times_conformal_factor, n_dot_longitudinal_shift_excess,
    inv_conformal_metric, conformal_christoffel_second_kind):
    x -= center
    make_metric_riemannian(inv_conformal_metric)
    conformal_unit_normal = -make_spherical_face_normal(
        x, inv_conformal_metric)
    local_projected_normal_gradient = projected_normal_gradient(
        conformal_unit_normal, x, inv_conformal_metric,
        conformal_christoffel_second_kind)
    lapse = lapse_times_conformal_factor / conformal_factor
    n_dot_longitudinal_shift = n_dot_longitudinal_shift_excess + np.einsum(
        'i,ij', -conformal_unit_normal, longitudinal_shift_background)
    J = (2. / 3. * extrinsic_curvature_trace - 0.5 / lapse *
         np.einsum('i,i', -conformal_unit_normal, n_dot_longitudinal_shift))
    J_correction = -0.5 * (
        (conformal_factor_correction / lapse_times_conformal_factor -
         conformal_factor / lapse_times_conformal_factor**2 *
         lapse_times_conformal_factor_correction) *
        np.einsum('i,i', -conformal_unit_normal, n_dot_longitudinal_shift) +
        1. / lapse * np.einsum('i,i', -conformal_unit_normal,
                               n_dot_longitudinal_shift_excess_correction))
    return 0.25 * (
        conformal_factor_correction * local_projected_normal_gradient -
        3. * conformal_factor**2 * conformal_factor_correction * J -
        conformal_factor**3 * J_correction)


def normal_dot_conformal_factor_gradient_correction_flat_cartesian(
    *args, **kwargs):
    return normal_dot_conformal_factor_gradient_correction(
        *args,
        inv_conformal_metric=np.identity(3),
        conformal_christoffel_second_kind=np.zeros((3, 3, 3)),
        **kwargs)


# This is the homogeneous-Neumann boundary condition on the lapse
def normal_dot_lapse_times_conformal_factor_gradient(*args, **kwargs):
    return 0.


# This function implements the quasi-equilibrium condition on the shift, i.e.
# Eq. 7.14 in Harald's thesis https://arxiv.org/abs/gr-qc/0510016
def shift_excess(conformal_factor, lapse_times_conformal_factor,
                 n_dot_longitudinal_shift_excess, center, spin, x,
                 extrinsic_curvature_trace, shift_background,
                 longitudinal_shift_background, inv_conformal_metric,
                 conformal_christoffel_second_kind):
    x -= center
    make_metric_riemannian(inv_conformal_metric)
    conformal_unit_normal = -make_spherical_face_normal(
        x, inv_conformal_metric)
    conformal_unit_normal_raised = np.einsum('ij,j', inv_conformal_metric,
                                             conformal_unit_normal)
    shift_orthogonal = lapse_times_conformal_factor / conformal_factor**3
    shift_parallel = np.cross(spin, x)
    return (shift_orthogonal * conformal_unit_normal_raised + shift_parallel -
            shift_background)


def shift_excess_flat_cartesian(*args, **kwargs):
    return shift_excess(*args,
                        inv_conformal_metric=np.identity(3),
                        conformal_christoffel_second_kind=np.zeros((3, 3, 3)),
                        **kwargs)


def shift_excess_correction(
    conformal_factor_correction, lapse_times_conformal_factor_correction,
    n_dot_longitudinal_shift_excess_correction, center, spin, x,
    extrinsic_curvature_trace, longitudinal_shift_background, conformal_factor,
    lapse_times_conformal_factor, n_dot_longitudinal_shift_excess,
    inv_conformal_metric, conformal_christoffel_second_kind):
    x -= center
    make_metric_riemannian(inv_conformal_metric)
    conformal_unit_normal = -make_spherical_face_normal(
        x, inv_conformal_metric)
    conformal_unit_normal_raised = np.einsum('ij,j', inv_conformal_metric,
                                             conformal_unit_normal)
    shift_orthogonal_correction = (
        lapse_times_conformal_factor_correction / conformal_factor**3 -
        3. * lapse_times_conformal_factor / conformal_factor**4 *
        conformal_factor_correction)
    return shift_orthogonal_correction * conformal_unit_normal_raised


def shift_excess_correction_flat_cartesian(*args, **kwargs):
    return shift_excess_correction(*args,
                                   inv_conformal_metric=np.identity(3),
                                   conformal_christoffel_second_kind=np.zeros(
                                       (3, 3, 3)),
                                   **kwargs)
