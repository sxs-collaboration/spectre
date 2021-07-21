# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import Evolution.Systems.GeneralizedHarmonic.TestFunctions as ght
import Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus as bjh


def error(face_mesh_velocity, normal_covector, normal_vector, gamma1, gamma2,
          lapse, shift, inverse_spacetime_metric, dt_spacetime_metric, dt_pi,
          dt_phi):
    # get char speeds
    char_speeds = [
        ght.char_speed_upsi(gamma1, lapse, shift, normal_covector),
        ght.char_speed_uzero(gamma1, lapse, shift, normal_covector),
        ght.char_speed_uplus(gamma1, lapse, shift, normal_covector),
        ght.char_speed_uminus(gamma1, lapse, shift, normal_covector)
    ]
    if face_mesh_velocity is not None:
        char_speeds = char_speeds - np.dot(normal_covector, face_mesh_velocity)
        if (np.dot(face_mesh_velocity, normal_covector) > 0.0):
            return (
                "We found the radial mesh velocity points in the direction "
                "of the outward normal, i.e. we possibly have an expanding "
                "domain. Its unclear if proper boundary conditions are "
                "imposed in this case.")
    return None


def dt_corrs_Freezing(face_mesh_velocity, normal_covector, normal_vector,
                      gamma1, gamma2, lapse, shift, inverse_spacetime_metric,
                      dt_spacetime_metric, dt_pi, dt_phi):
    # get char speeds
    char_speeds = [
        ght.char_speed_upsi(gamma1, lapse, shift, normal_covector),
        ght.char_speed_uzero(gamma1, lapse, shift, normal_covector),
        ght.char_speed_uplus(gamma1, lapse, shift, normal_covector),
        ght.char_speed_uminus(gamma1, lapse, shift, normal_covector)
    ]
    if face_mesh_velocity is not None:
        char_speeds = char_speeds - np.dot(normal_covector, face_mesh_velocity)
    # get char projections of dt<RHS>
    inverse_spatial_metric = (inverse_spacetime_metric[1:, 1:] +
                              (np.einsum('i,j->ij', shift, shift) /
                               (lapse * lapse)))
    char_projected_rhs_dt_v_psi = ght.char_field_upsi(gamma2,
                                                      inverse_spatial_metric,
                                                      dt_spacetime_metric,
                                                      dt_pi, dt_phi,
                                                      normal_covector)
    char_projected_rhs_dt_v_zero = ght.char_field_uzero(
        gamma2, inverse_spatial_metric, dt_spacetime_metric, dt_pi, dt_phi,
        normal_covector)
    char_projected_rhs_dt_v_plus = ght.char_field_uplus(
        gamma2, inverse_spatial_metric, dt_spacetime_metric, dt_pi, dt_phi,
        normal_covector)
    char_projected_rhs_dt_v_minus = ght.char_field_uminus(
        gamma2, inverse_spatial_metric, dt_spacetime_metric, dt_pi, dt_phi,
        normal_covector)
    # freezing corrections
    dt_v_psi = -1 * char_projected_rhs_dt_v_psi
    dt_v_zero = -1 * char_projected_rhs_dt_v_zero
    dt_v_plus = -1 * char_projected_rhs_dt_v_plus
    dt_v_minus = -1 * char_projected_rhs_dt_v_minus
    # set only if incoming
    dt_v_psi = bjh.set_bc_corr_zero_when_char_speed_is_positive(
        dt_v_psi, char_speeds[0])
    dt_v_zero = bjh.set_bc_corr_zero_when_char_speed_is_positive(
        dt_v_zero, char_speeds[1])
    dt_v_plus = bjh.set_bc_corr_zero_when_char_speed_is_positive(
        dt_v_plus, char_speeds[2])
    dt_v_minus = bjh.set_bc_corr_zero_when_char_speed_is_positive(
        dt_v_minus, char_speeds[3])
    return (dt_v_psi, dt_v_zero, dt_v_plus, dt_v_minus)


def dt_spacetime_metric_Freezing(face_mesh_velocity, normal_covector,
                                 normal_vector, gamma1, gamma2, lapse, shift,
                                 inverse_spacetime_metric, dt_spacetime_metric,
                                 dt_pi, dt_phi):
    (dt_v_psi, _, _, _) = dt_corrs_Freezing(face_mesh_velocity,
                                            normal_covector, normal_vector,
                                            gamma1, gamma2, lapse, shift,
                                            inverse_spacetime_metric,
                                            dt_spacetime_metric, dt_pi, dt_phi)
    return dt_v_psi


def dt_pi_Freezing(face_mesh_velocity, normal_covector, normal_vector, gamma1,
                   gamma2, lapse, shift, inverse_spacetime_metric,
                   dt_spacetime_metric, dt_pi, dt_phi):
    (dt_v_psi, dt_v_zero, dt_v_plus,
     dt_v_minus) = dt_corrs_Freezing(face_mesh_velocity, normal_covector,
                                     normal_vector, gamma1, gamma2, lapse,
                                     shift, inverse_spacetime_metric,
                                     dt_spacetime_metric, dt_pi, dt_phi)
    return ght.evol_field_pi(gamma2, dt_v_psi, dt_v_zero, dt_v_plus,
                             dt_v_minus, normal_covector)


def dt_phi_Freezing(face_mesh_velocity, normal_covector, normal_vector, gamma1,
                    gamma2, lapse, shift, inverse_spacetime_metric,
                    dt_spacetime_metric, dt_pi, dt_phi):
    (dt_v_psi, dt_v_zero, dt_v_plus,
     dt_v_minus) = dt_corrs_Freezing(face_mesh_velocity, normal_covector,
                                     normal_vector, gamma1, gamma2, lapse,
                                     shift, inverse_spacetime_metric,
                                     dt_spacetime_metric, dt_pi, dt_phi)
    return ght.evol_field_phi(gamma2, dt_v_psi, dt_v_zero, dt_v_plus,
                              dt_v_minus, normal_covector)
