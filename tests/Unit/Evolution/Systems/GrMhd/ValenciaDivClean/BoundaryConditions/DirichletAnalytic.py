# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import PointwiseFunctions.AnalyticSolutions.Hydro.SmoothFlow as hydro
import Evolution.Systems.GrMhd.ValenciaDivClean.TestFunctions as cons
import Evolution.Systems.GrMhd.ValenciaDivClean.Fluxes as fluxes
import PointwiseFunctions.AnalyticData.GrMhd.MagneticRotor as rotor


def soln_error(face_mesh_velocity, outward_directed_normal_covector,
               outward_directed_normal_vector, coords, time, dim):
    return None


_soln_pressure = 1.0
_soln_adiabatic_index = 5.0 / 3.0
_soln_perturbation_size = 0.2


def _soln_mean_velocity():
    dim = 3
    mean_v = []
    for i in range(0, dim):
        mean_v.append(0.9 - i * 0.5)
    return np.asarray(mean_v)


def _soln_wave_vector():
    dim = 3
    wave_vector = []
    for i in range(0, dim):
        wave_vector.append(0.1 + i)
    return np.asarray(wave_vector)


def soln_velocity(coords, time):
    return hydro.spatial_velocity(coords, time, _soln_mean_velocity(),
                                  _soln_wave_vector(), _soln_pressure,
                                  _soln_adiabatic_index,
                                  _soln_perturbation_size)


def soln_lorentz_factor(coords, time):
    return hydro.lorentz_factor(coords, time, _soln_mean_velocity(),
                                _soln_wave_vector(), _soln_pressure,
                                _soln_adiabatic_index, _soln_perturbation_size)


def soln_specific_internal_energy(coords, time):
    return hydro.specific_internal_energy(coords, time, _soln_mean_velocity(),
                                          _soln_wave_vector(), _soln_pressure,
                                          _soln_adiabatic_index,
                                          _soln_perturbation_size)


def soln_specific_enthalpy(coords, time):
    return hydro.specific_enthalpy_relativistic(coords, time,
                                                _soln_mean_velocity(),
                                                _soln_wave_vector(),
                                                _soln_pressure,
                                                _soln_adiabatic_index,
                                                _soln_perturbation_size)


def soln_pressure(coords, time):
    return _soln_pressure


def soln_mass_density(coords, time):
    return hydro.rest_mass_density(coords, time, _soln_mean_velocity(),
                                   _soln_wave_vector(), _soln_pressure,
                                   _soln_adiabatic_index,
                                   _soln_perturbation_size)


def soln_magnetic_field(coords, time):
    return np.array([0.0, 0.0, 0.0])


def divergence_cleaning_field(coords, time):
    return 0.0


def soln_sqrt_det_spatial_metric(coords, time):
    return 1.0


def soln_spatial_metric(coords, time):
    return np.identity(3)


def soln_inverse_spatial_metric(coords, time):
    return np.identity(3)


def soln_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                 outward_directed_normal_vector, coords, time, dim):
    return soln_lorentz_factor(coords, time) * soln_mass_density(coords, time)


def soln_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim):
    return cons.tilde_tau(soln_mass_density(coords, time),
                          soln_specific_internal_energy(coords, time),
                          soln_specific_enthalpy(coords, time),
                          soln_pressure(coords, time),
                          soln_velocity(coords, time),
                          soln_lorentz_factor(coords, time),
                          soln_magnetic_field(coords, time),
                          soln_sqrt_det_spatial_metric(coords, time),
                          soln_spatial_metric(coords, time),
                          divergence_cleaning_field(coords, time))


def soln_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                 outward_directed_normal_vector, coords, time, dim):
    return cons.tilde_s(soln_mass_density(coords, time),
                        soln_specific_internal_energy(coords, time),
                        soln_specific_enthalpy(coords, time),
                        soln_pressure(coords, time),
                        soln_velocity(coords, time),
                        soln_lorentz_factor(coords, time),
                        soln_magnetic_field(coords, time),
                        soln_sqrt_det_spatial_metric(coords, time),
                        soln_spatial_metric(coords, time),
                        divergence_cleaning_field(coords, time))


def soln_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                 outward_directed_normal_vector, coords, time, dim):
    return cons.tilde_b(soln_mass_density(coords, time),
                        soln_specific_internal_energy(coords, time),
                        soln_specific_enthalpy(coords, time),
                        soln_pressure(coords, time),
                        soln_velocity(coords, time),
                        soln_lorentz_factor(coords, time),
                        soln_magnetic_field(coords, time),
                        soln_sqrt_det_spatial_metric(coords, time),
                        soln_spatial_metric(coords, time),
                        divergence_cleaning_field(coords, time))


def soln_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim):
    return cons.tilde_phi(soln_mass_density(coords, time),
                          soln_specific_internal_energy(coords, time),
                          soln_specific_enthalpy(coords, time),
                          soln_pressure(coords, time),
                          soln_velocity(coords, time),
                          soln_lorentz_factor(coords, time),
                          soln_magnetic_field(coords, time),
                          soln_sqrt_det_spatial_metric(coords, time),
                          soln_spatial_metric(coords, time),
                          divergence_cleaning_field(coords, time))


def soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
               outward_directed_normal_vector, coords, time, dim):
    return 1.0


def soln_shift(face_mesh_velocity, outward_directed_normal_covector,
               outward_directed_normal_vector, coords, time, dim):
    return np.asarray([0.0, 0.0, 0.0])


def soln_flux_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_d_flux(
        soln_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), soln_pressure(coords, time),
        soln_velocity(coords, time), soln_lorentz_factor(coords, time),
        soln_magnetic_field(coords, time))


def soln_flux_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                        outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_tau_flux(
        soln_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), soln_pressure(coords, time),
        soln_velocity(coords, time), soln_lorentz_factor(coords, time),
        soln_magnetic_field(coords, time))


def soln_flux_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_s_flux(
        soln_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), soln_pressure(coords, time),
        soln_velocity(coords, time), soln_lorentz_factor(coords, time),
        soln_magnetic_field(coords, time))


def soln_flux_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_b_flux(
        soln_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), soln_pressure(coords, time),
        soln_velocity(coords, time), soln_lorentz_factor(coords, time),
        soln_magnetic_field(coords, time))


def soln_flux_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                        outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_phi_flux(
        soln_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        soln_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), soln_pressure(coords, time),
        soln_velocity(coords, time), soln_lorentz_factor(coords, time),
        soln_magnetic_field(coords, time))


_data_rotor_radius = 0.1
_data_rotor_density = 10.0
_data_background_density = 1.0
_data_pressure = 1.0
_data_angular_velocity = 9.95
_data_adiabatic_index = 5.0 / 3.0
_data_magnetic_field = np.asarray([3.54490770181103205, 0.0, 0.0])


def data_velocity(coords):
    return rotor.spatial_velocity(coords, _data_rotor_radius,
                                  _data_rotor_density,
                                  _data_background_density, _data_pressure,
                                  _data_angular_velocity, _data_magnetic_field,
                                  _data_adiabatic_index)


def data_lorentz_factor(coords):
    return rotor.lorentz_factor(coords, _data_rotor_radius,
                                _data_rotor_density, _data_background_density,
                                _data_pressure, _data_angular_velocity,
                                _data_magnetic_field, _data_adiabatic_index)


def data_specific_internal_energy(coords):
    return rotor.specific_internal_energy(
        coords, _data_rotor_radius, _data_rotor_density,
        _data_background_density, _data_pressure, _data_angular_velocity,
        _data_magnetic_field, _data_adiabatic_index)


def data_specific_enthalpy(coords):
    return rotor.specific_enthalpy(coords, _data_rotor_radius,
                                   _data_rotor_density,
                                   _data_background_density, _data_pressure,
                                   _data_angular_velocity,
                                   _data_magnetic_field, _data_adiabatic_index)


def data_pressure(coords):
    return _data_pressure


def data_mass_density(coords):
    return rotor.rest_mass_density(coords, _data_rotor_radius,
                                   _data_rotor_density,
                                   _data_background_density, _data_pressure,
                                   _data_angular_velocity,
                                   _data_magnetic_field, _data_adiabatic_index)


def data_magnetic_field(coords):
    return rotor.magnetic_field(coords, _data_rotor_radius,
                                _data_rotor_density, _data_background_density,
                                _data_pressure, _data_angular_velocity,
                                _data_magnetic_field, _data_adiabatic_index)


def data_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                 outward_directed_normal_vector, coords, time, dim):
    return data_lorentz_factor(coords) * data_mass_density(coords)


def data_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim):
    return cons.tilde_tau(data_mass_density(coords),
                          data_specific_internal_energy(coords),
                          data_specific_enthalpy(coords),
                          data_pressure(coords), data_velocity(coords),
                          data_lorentz_factor(coords),
                          data_magnetic_field(coords),
                          soln_sqrt_det_spatial_metric(coords, time),
                          soln_spatial_metric(coords, time),
                          divergence_cleaning_field(coords, time))


def data_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                 outward_directed_normal_vector, coords, time, dim):
    return cons.tilde_s(data_mass_density(coords),
                        data_specific_internal_energy(coords),
                        data_specific_enthalpy(coords), data_pressure(coords),
                        data_velocity(coords), data_lorentz_factor(coords),
                        data_magnetic_field(coords),
                        soln_sqrt_det_spatial_metric(coords, time),
                        soln_spatial_metric(coords, time),
                        divergence_cleaning_field(coords, time))


def data_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                 outward_directed_normal_vector, coords, time, dim):
    return cons.tilde_b(data_mass_density(coords),
                        data_specific_internal_energy(coords),
                        data_specific_enthalpy(coords), data_pressure(coords),
                        data_velocity(coords), data_lorentz_factor(coords),
                        data_magnetic_field(coords),
                        soln_sqrt_det_spatial_metric(coords, time),
                        soln_spatial_metric(coords, time),
                        divergence_cleaning_field(coords, time))


def data_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim):
    return cons.tilde_phi(data_mass_density(coords),
                          data_specific_internal_energy(coords),
                          data_specific_enthalpy(coords),
                          data_pressure(coords), data_velocity(coords),
                          data_lorentz_factor(coords),
                          data_magnetic_field(coords),
                          soln_sqrt_det_spatial_metric(coords, time),
                          soln_spatial_metric(coords, time),
                          divergence_cleaning_field(coords, time))


def data_flux_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_d_flux(
        data_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        data_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), data_pressure(coords),
        data_velocity(coords), data_lorentz_factor(coords),
        data_magnetic_field(coords))


def data_flux_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                        outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_tau_flux(
        data_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        data_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), data_pressure(coords),
        data_velocity(coords), data_lorentz_factor(coords),
        data_magnetic_field(coords))


def data_flux_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_s_flux(
        data_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        data_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), data_pressure(coords),
        data_velocity(coords), data_lorentz_factor(coords),
        data_magnetic_field(coords))


def data_flux_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_b_flux(
        data_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        data_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), data_pressure(coords),
        data_velocity(coords), data_lorentz_factor(coords),
        data_magnetic_field(coords))


def data_flux_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                        outward_directed_normal_vector, coords, time, dim):
    return fluxes.tilde_phi_flux(
        data_tilde_d(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_tau(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        data_tilde_s(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_b(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, time, dim),
        data_tilde_phi(face_mesh_velocity, outward_directed_normal_covector,
                       outward_directed_normal_vector, coords, time, dim),
        soln_lapse(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_shift(face_mesh_velocity, outward_directed_normal_covector,
                   outward_directed_normal_vector, coords, time, dim),
        soln_sqrt_det_spatial_metric(coords, time),
        soln_spatial_metric(coords, time),
        soln_inverse_spatial_metric(coords, time), data_pressure(coords),
        data_velocity(coords), data_lorentz_factor(coords),
        data_magnetic_field(coords))
