# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import PointwiseFunctions.AnalyticSolutions.RadiationTransport.M1Grey.\
           TestFunctions as soln
import Evolution.Systems.RadiationTransport.M1Grey.Fluxes as fluxes


def soln_error(face_mesh_velocity, outward_directed_normal_covector,
               outward_directed_normal_vector, coords, time, dim):
    return None


_soln_mean_velocity = np.array([0.1, 0.2, 0.3])
_soln_comoving_energy_density = 0.4

# There is no Python implementation of the M1Closure, so for now we hardcode
# the pressure tensor with values obtained by running the C++ code run with
# the same input parameters. This is hacky, because it couples the C++ and
# Python implementations and makes the test less robust.
_tilde_p_values_from_cxx = np.array(
    [[0.13953488372093026, 0.012403100775193798, 0.018604651162790694],
     [0.012403100775193798, 0.15813953488372096, 0.037209302325581388],
     [0.018604651162790694, 0.037209302325581388, 0.18914728682170545]])


def soln_tilde_e_nue(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim):
    return soln.constant_m1_tildeE(coords, time, _soln_mean_velocity,
                                   _soln_comoving_energy_density)


def soln_tilde_e_bar_nue(face_mesh_velocity, outward_directed_normal_covector,
                         outward_directed_normal_vector, coords, time, dim):
    return soln.constant_m1_tildeE(coords, time, _soln_mean_velocity,
                                   _soln_comoving_energy_density)


def soln_tilde_s_nue(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim):
    return soln.constant_m1_tildeS(coords, time, _soln_mean_velocity,
                                   _soln_comoving_energy_density)


def soln_tilde_s_bar_nue(face_mesh_velocity, outward_directed_normal_covector,
                         outward_directed_normal_vector, coords, time, dim):
    return soln.constant_m1_tildeS(coords, time, _soln_mean_velocity,
                                   _soln_comoving_energy_density)


def soln_flux_tilde_e_nue(face_mesh_velocity, outward_directed_normal_covector,
                          outward_directed_normal_vector, coords, time, dim):
    tilde_e = soln_tilde_e_nue(face_mesh_velocity,
                               outward_directed_normal_covector,
                               outward_directed_normal_vector, coords, time,
                               dim)
    tilde_s = soln_tilde_s_nue(face_mesh_velocity,
                               outward_directed_normal_covector,
                               outward_directed_normal_vector, coords, time,
                               dim)
    tilde_p = _tilde_p_values_from_cxx
    lapse = 1.0
    shift = np.array([0.0, 0.0, 0.0])
    spatial_metric = np.identity(3)
    inv_spatial_metric = np.identity(3)
    return fluxes.tilde_e_flux(tilde_e, tilde_s, tilde_p, lapse, shift,
                               spatial_metric, inv_spatial_metric)


def soln_flux_tilde_e_bar_nue(face_mesh_velocity,
                              outward_directed_normal_covector,
                              outward_directed_normal_vector, coords, time,
                              dim):
    tilde_e = soln_tilde_e_bar_nue(face_mesh_velocity,
                                   outward_directed_normal_covector,
                                   outward_directed_normal_vector, coords,
                                   time, dim)
    tilde_s = soln_tilde_s_bar_nue(face_mesh_velocity,
                                   outward_directed_normal_covector,
                                   outward_directed_normal_vector, coords,
                                   time, dim)
    tilde_p = _tilde_p_values_from_cxx
    lapse = 1.0
    shift = np.array([0.0, 0.0, 0.0])
    spatial_metric = np.identity(3)
    inv_spatial_metric = np.identity(3)
    return fluxes.tilde_e_flux(tilde_e, tilde_s, tilde_p, lapse, shift,
                               spatial_metric, inv_spatial_metric)


def soln_flux_tilde_s_nue(face_mesh_velocity, outward_directed_normal_covector,
                          outward_directed_normal_vector, coords, time, dim):
    tilde_e = soln_tilde_e_nue(face_mesh_velocity,
                               outward_directed_normal_covector,
                               outward_directed_normal_vector, coords, time,
                               dim)
    tilde_s = soln_tilde_s_nue(face_mesh_velocity,
                               outward_directed_normal_covector,
                               outward_directed_normal_vector, coords, time,
                               dim)
    tilde_p = _tilde_p_values_from_cxx
    lapse = 1.0
    shift = np.array([0.0, 0.0, 0.0])
    spatial_metric = np.identity(3)
    inv_spatial_metric = np.identity(3)
    return fluxes.tilde_s_flux(tilde_e, tilde_s, tilde_p, lapse, shift,
                               spatial_metric, inv_spatial_metric)


def soln_flux_tilde_s_bar_nue(face_mesh_velocity,
                              outward_directed_normal_covector,
                              outward_directed_normal_vector, coords, time,
                              dim):
    tilde_e = soln_tilde_e_bar_nue(face_mesh_velocity,
                                   outward_directed_normal_covector,
                                   outward_directed_normal_vector, coords,
                                   time, dim)
    tilde_s = soln_tilde_s_bar_nue(face_mesh_velocity,
                                   outward_directed_normal_covector,
                                   outward_directed_normal_vector, coords,
                                   time, dim)
    tilde_p = _tilde_p_values_from_cxx
    lapse = 1.0
    shift = np.array([0.0, 0.0, 0.0])
    spatial_metric = np.identity(3)
    inv_spatial_metric = np.identity(3)
    return fluxes.tilde_s_flux(tilde_e, tilde_s, tilde_p, lapse, shift,
                               spatial_metric, inv_spatial_metric)
