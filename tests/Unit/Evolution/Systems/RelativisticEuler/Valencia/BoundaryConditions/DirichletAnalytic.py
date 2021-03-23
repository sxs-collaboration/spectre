# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import PointwiseFunctions.AnalyticSolutions.Hydro.SmoothFlow as hydro
import Evolution.Systems.RelativisticEuler.Valencia.Fluxes as flux


def soln_error(face_mesh_velocity, outward_directed_normal_covector,
               outward_directed_normal_vector, coords, time, dim):
    return None


_soln_pressure = 1.0
_soln_adiabatic_index = 5.0 / 3.0
_soln_perturbation_size = 0.2


def _soln_mean_velocity(dim):
    mean_v = []
    for i in range(0, dim):
        mean_v.append(0.9 - i * 0.5)
    return np.asarray(mean_v)


def _soln_wave_vector(dim):
    wave_vector = []
    for i in range(0, dim):
        wave_vector.append(0.1 + i)
    return np.asarray(wave_vector)


def soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
               outward_directed_normal_vector, coords, time, dim):
    return 1.0


def soln_shift(face_mesh_velocity, outward_directed_normal_covector,
               outward_directed_normal_vector, coords, time, dim):
    return np.zeros(dim)


def soln_spatial_metric(face_mesh_velocity, outward_directed_normal_covector,
                        outward_directed_normal_vector, coords, time, dim):
    return np.identity(dim)


def soln_rest_mass_density(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim):
    return hydro.rest_mass_density(coords, time, _soln_mean_velocity(dim),
                                   _soln_wave_vector(dim), _soln_pressure,
                                   _soln_adiabatic_index,
                                   _soln_perturbation_size)


def soln_specific_internal_energy(face_mesh_velocity,
                                  outward_directed_normal_covector,
                                  outward_directed_normal_vector, coords, time,
                                  dim):
    return hydro.specific_internal_energy(coords, time,
                                          _soln_mean_velocity(dim),
                                          _soln_wave_vector(dim),
                                          _soln_pressure,
                                          _soln_adiabatic_index,
                                          _soln_perturbation_size)


def soln_specific_enthalpy(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim):
    return hydro.specific_enthalpy_relativistic(coords, time,
                                                _soln_mean_velocity(dim),
                                                _soln_wave_vector(dim),
                                                _soln_pressure,
                                                _soln_adiabatic_index,
                                                _soln_perturbation_size)


def soln_spatial_velocity(face_mesh_velocity, outward_directed_normal_covector,
                          outward_directed_normal_vector, coords, time, dim):
    return hydro.spatial_velocity(coords, time, _soln_mean_velocity(dim),
                                  _soln_wave_vector(dim), _soln_pressure,
                                  _soln_adiabatic_index,
                                  _soln_perturbation_size)


def soln_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                 outward_directed_normal_vector, coords, time, dim):
    rho = soln_rest_mass_density(face_mesh_velocity,
                                 outward_directed_normal_covector,
                                 outward_directed_normal_vector, coords, time,
                                 dim)
    W = hydro.lorentz_factor(coords, time, _soln_mean_velocity(dim),
                             _soln_wave_vector(dim), _soln_pressure,
                             _soln_adiabatic_index, _soln_perturbation_size)
    # for smooth flow, sqrt_det_spatial_metric = 1
    return rho * W


def soln_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim):
    rho = soln_rest_mass_density(face_mesh_velocity,
                                 outward_directed_normal_covector,
                                 outward_directed_normal_vector, coords, time,
                                 dim)
    eps = soln_specific_internal_energy(face_mesh_velocity,
                                        outward_directed_normal_covector,
                                        outward_directed_normal_vector, coords,
                                        time, dim)
    v = soln_spatial_velocity(face_mesh_velocity,
                              outward_directed_normal_covector,
                              outward_directed_normal_vector, coords, time,
                              dim)
    v2 = v.dot(v)
    W = hydro.lorentz_factor(coords, time, _soln_mean_velocity(dim),
                             _soln_wave_vector(dim), _soln_pressure,
                             _soln_adiabatic_index, _soln_perturbation_size)
    # for smooth flow, sqrt_det_spatial_metric = 1
    return W**2 * (rho * (eps + v2 * W / (W + 1.0)) + _soln_pressure * v2)


def soln_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                 outward_directed_normal_vector, coords, time, dim):
    tilde_d = soln_tilde_d(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim)
    W = hydro.lorentz_factor(coords, time, _soln_mean_velocity(dim),
                             _soln_wave_vector(dim), _soln_pressure,
                             _soln_adiabatic_index, _soln_perturbation_size)
    h = soln_specific_enthalpy(face_mesh_velocity,
                               outward_directed_normal_covector,
                               outward_directed_normal_vector, coords, time,
                               dim)
    # for smooth flow, metric = identity ==> v_lower = v_upper
    v_lower = soln_spatial_velocity(face_mesh_velocity,
                                    outward_directed_normal_covector,
                                    outward_directed_normal_vector, coords,
                                    time, dim)
    return tilde_d * W * h * v_lower


def soln_flux_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, time, dim):
    tilde_d = soln_tilde_d(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim)
    tilde_tau = soln_tilde_tau(face_mesh_velocity,
                               outward_directed_normal_covector,
                               outward_directed_normal_vector, coords, time,
                               dim)
    tilde_s = soln_tilde_s(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim)
    lapse = soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim)
    shift = soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim)
    sqrt_det_spatial_metric = 1.0
    pressure = _soln_pressure
    spatial_velocity = soln_spatial_velocity(face_mesh_velocity,
                                             outward_directed_normal_covector,
                                             outward_directed_normal_vector,
                                             coords, time, dim)
    return flux.tilde_d_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                             sqrt_det_spatial_metric, pressure,
                             spatial_velocity)


def soln_flux_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                        outward_directed_normal_vector, coords, time, dim):
    tilde_d = soln_tilde_d(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim)
    tilde_tau = soln_tilde_tau(face_mesh_velocity,
                               outward_directed_normal_covector,
                               outward_directed_normal_vector, coords, time,
                               dim)
    tilde_s = soln_tilde_s(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim)
    lapse = soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim)
    shift = soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim)
    sqrt_det_spatial_metric = 1.0
    pressure = _soln_pressure
    spatial_velocity = soln_spatial_velocity(face_mesh_velocity,
                                             outward_directed_normal_covector,
                                             outward_directed_normal_vector,
                                             coords, time, dim)
    return flux.tilde_tau_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                               sqrt_det_spatial_metric, pressure,
                               spatial_velocity)


def soln_flux_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, time, dim):
    tilde_d = soln_tilde_d(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim)
    tilde_tau = soln_tilde_tau(face_mesh_velocity,
                               outward_directed_normal_covector,
                               outward_directed_normal_vector, coords, time,
                               dim)
    tilde_s = soln_tilde_s(face_mesh_velocity,
                           outward_directed_normal_covector,
                           outward_directed_normal_vector, coords, time, dim)
    lapse = soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim)
    shift = soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim)
    sqrt_det_spatial_metric = 1.0
    pressure = _soln_pressure
    spatial_velocity = soln_spatial_velocity(face_mesh_velocity,
                                             outward_directed_normal_covector,
                                             outward_directed_normal_vector,
                                             coords, time, dim)
    return flux.tilde_s_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                             sqrt_det_spatial_metric, pressure,
                             spatial_velocity)
