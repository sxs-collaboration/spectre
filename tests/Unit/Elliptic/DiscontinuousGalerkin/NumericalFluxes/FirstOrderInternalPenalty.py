# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def penalty(element_size, perpendicular_num_points, penalty_parameter):
    return penalty_parameter * perpendicular_num_points**2 / element_size


def average(int, ext):
    return 0.5 * (int + ext)


def jump(int, ext):
    return int - ext


def flux_for_field(grad_field, fluxes_argument):
    return fluxes_argument * np.asarray(grad_field)


def flux_for_auxiliary_field(field, fluxes_argument, dim):
    return np.diag(np.repeat(fluxes_argument * field, dim))


def normal_dot_numerical_flux_for_field(n_dot_aux_flux_int, n_dot_aux_flux_ext,
                                        div_aux_flux_int, div_aux_flux_ext,
                                        fluxes_argument, penalty_parameter,
                                        face_normal_magnitude_int,
                                        face_normal_magnitude_ext,
                                        face_normal_int):
    sigma = penalty(
        np.minimum(2. / face_normal_magnitude_int,
                   2. / face_normal_magnitude_ext), 4, penalty_parameter)
    return np.dot(
        face_normal_int,
        average(flux_for_field(div_aux_flux_int, fluxes_argument),
                flux_for_field(div_aux_flux_ext, fluxes_argument)) -
        sigma * jump(flux_for_field(n_dot_aux_flux_int, fluxes_argument),
                     flux_for_field(-n_dot_aux_flux_ext, fluxes_argument)))


def normal_dot_numerical_flux_for_auxiliary_field(
    n_dot_aux_flux_int, minus_n_dot_aux_flux_ext, div_aux_flux_int,
    div_aux_flux_ext, fluxes_argument, penalty_parameter,
    face_normal_magnitude_int, face_normal_magnitude_ext, face_normal_int):
    # `minus_n_dot_aux_flux_ext` is from the element on the other side of the
    # interface, so it is _minus_ the same quantity if it was computed with the
    # normal from this element (assuming normals don't depend on the dynamic
    # variables). See the `()` operator in
    # `elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty` for details.
    return average(n_dot_aux_flux_int, -minus_n_dot_aux_flux_ext)


def normal_dot_dirichlet_flux_for_field(dirichlet_field, fluxes_argument,
                                        penalty_parameter,
                                        face_normal_magnitude, face_normal,
                                        dim):
    sigma = penalty(2. / face_normal_magnitude, 3, penalty_parameter)
    return 2. * sigma * np.dot(
        face_normal,
        flux_for_field(
            np.dot(
                face_normal,
                flux_for_auxiliary_field(dirichlet_field, fluxes_argument,
                                         dim)), fluxes_argument))


def normal_dot_dirichlet_flux_for_field_1d(*args, **kwargs):
    return normal_dot_dirichlet_flux_for_field(*args, dim=1, **kwargs)


def normal_dot_dirichlet_flux_for_field_2d(*args, **kwargs):
    return normal_dot_dirichlet_flux_for_field(*args, dim=2, **kwargs)


def normal_dot_dirichlet_flux_for_field_3d(*args, **kwargs):
    return normal_dot_dirichlet_flux_for_field(*args, dim=3, **kwargs)


def normal_dot_dirichlet_flux_for_auxiliary_field(dirichlet_field,
                                                  fluxes_argument,
                                                  penalty_parameter,
                                                  face_normal_magnitude,
                                                  face_normal, dim):
    return np.dot(
        face_normal,
        flux_for_auxiliary_field(dirichlet_field, fluxes_argument, dim))


def normal_dot_dirichlet_flux_for_auxiliary_field_1d(*args, **kwargs):
    return normal_dot_dirichlet_flux_for_auxiliary_field(*args,
                                                         dim=1,
                                                         **kwargs)


def normal_dot_dirichlet_flux_for_auxiliary_field_2d(*args, **kwargs):
    return normal_dot_dirichlet_flux_for_auxiliary_field(*args,
                                                         dim=2,
                                                         **kwargs)


def normal_dot_dirichlet_flux_for_auxiliary_field_3d(*args, **kwargs):
    return normal_dot_dirichlet_flux_for_auxiliary_field(*args,
                                                         dim=3,
                                                         **kwargs)
