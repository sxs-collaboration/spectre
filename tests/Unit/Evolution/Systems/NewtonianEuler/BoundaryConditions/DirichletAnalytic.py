# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import PointwiseFunctions.AnalyticSolutions.Hydro.SmoothFlow as hydro
import Evolution.Systems.NewtonianEuler.TimeDerivative as flux


def soln_error(face_mesh_velocity, outward_directed_normal_covector, coords,
               time, dim):
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


def soln_mass_density(face_mesh_velocity, outward_directed_normal_covector,
                      coords, time, dim):
    return hydro.rest_mass_density(coords, time, _soln_mean_velocity(dim),
                                   _soln_wave_vector(dim), _soln_pressure,
                                   _soln_adiabatic_index,
                                   _soln_perturbation_size)


def soln_momentum_density(face_mesh_velocity, outward_directed_normal_covector,
                          coords, time, dim):
    return hydro.rest_mass_density(
        coords, time, _soln_mean_velocity(dim), _soln_wave_vector(dim),
        _soln_pressure, _soln_adiabatic_index,
        _soln_perturbation_size) * hydro.spatial_velocity(
            coords, time, _soln_mean_velocity(dim), _soln_wave_vector(dim),
            _soln_pressure, _soln_adiabatic_index, _soln_perturbation_size)


def soln_energy_density(face_mesh_velocity, outward_directed_normal_covector,
                        coords, time, dim):
    velocity = hydro.spatial_velocity(coords, time, _soln_mean_velocity(dim),
                                      _soln_wave_vector(dim), _soln_pressure,
                                      _soln_adiabatic_index,
                                      _soln_perturbation_size)
    int_energy = hydro.specific_internal_energy(coords, time,
                                                _soln_mean_velocity(dim),
                                                _soln_wave_vector(dim),
                                                _soln_pressure,
                                                _soln_adiabatic_index,
                                                _soln_perturbation_size)
    return hydro.rest_mass_density(
        coords, time, _soln_mean_velocity(dim), _soln_wave_vector(dim),
        _soln_pressure, _soln_adiabatic_index, _soln_perturbation_size) * (
            0.5 * np.dot(velocity, velocity) + int_energy)


def soln_flux_mass_density(face_mesh_velocity,
                           outward_directed_normal_covector, coords, time,
                           dim):
    return soln_momentum_density(face_mesh_velocity,
                                 outward_directed_normal_covector, coords,
                                 time, dim)


def soln_flux_momentum_density(face_mesh_velocity,
                               outward_directed_normal_covector, coords, time,
                               dim):
    velocity = hydro.spatial_velocity(coords, time, _soln_mean_velocity(dim),
                                      _soln_wave_vector(dim), _soln_pressure,
                                      _soln_adiabatic_index,
                                      _soln_perturbation_size)
    pressure = hydro.pressure(coords, time, _soln_mean_velocity(dim),
                              _soln_wave_vector(dim), _soln_pressure,
                              _soln_adiabatic_index, _soln_perturbation_size)
    return flux.momentum_density_flux_impl(
        soln_momentum_density(face_mesh_velocity,
                              outward_directed_normal_covector, coords, time,
                              dim),
        soln_energy_density(face_mesh_velocity,
                            outward_directed_normal_covector, coords, time,
                            dim), velocity, pressure)


def soln_flux_energy_density(face_mesh_velocity,
                             outward_directed_normal_covector, coords, time,
                             dim):
    velocity = hydro.spatial_velocity(coords, time, _soln_mean_velocity(dim),
                                      _soln_wave_vector(dim), _soln_pressure,
                                      _soln_adiabatic_index,
                                      _soln_perturbation_size)
    pressure = hydro.pressure(coords, time, _soln_mean_velocity(dim),
                              _soln_wave_vector(dim), _soln_pressure,
                              _soln_adiabatic_index, _soln_perturbation_size)
    return flux.energy_density_flux_impl(
        soln_momentum_density(face_mesh_velocity,
                              outward_directed_normal_covector, coords, time,
                              dim),
        soln_energy_density(face_mesh_velocity,
                            outward_directed_normal_covector, coords, time,
                            dim), velocity, pressure)


def soln_velocity(face_mesh_velocity, outward_directed_normal_covector, coords,
                  time, dim):
    return hydro.spatial_velocity(coords, time, _soln_mean_velocity(dim),
                                  _soln_wave_vector(dim), _soln_pressure,
                                  _soln_adiabatic_index,
                                  _soln_perturbation_size)


def soln_specific_internal_energy(face_mesh_velocity,
                                  outward_directed_normal_covector, coords,
                                  time, dim):
    return hydro.specific_internal_energy(coords, time,
                                          _soln_mean_velocity(dim),
                                          _soln_wave_vector(dim),
                                          _soln_pressure,
                                          _soln_adiabatic_index,
                                          _soln_perturbation_size)


def data_error(face_mesh_velocity, outward_directed_normal_covector, coords,
               time, dim):
    return None


_data_adiabatic_index = 1.4
_data_strip_bimedian_height = 0.5
_data_strip_thickness = 0.5
_data_strip_density = 2.0
_data_strip_velocity = 0.5
_data_background_density = 1.0
_data_background_velocity = -0.5
_data_pressure = 2.5
_data_perturb_amplitude = 0.1
_data_perturb_width = 0.03


def data_mass_density(face_mesh_velocity, outward_directed_normal_covector,
                      coords, time, dim):
    if np.abs(coords[-1] -
              _data_strip_bimedian_height) < 0.5 * _data_strip_thickness:
        return _data_strip_density
    else:
        return _data_background_density


def data_velocity(face_mesh_velocity, outward_directed_normal_covector, coords,
                  time, dim):
    velocity = np.zeros([dim])
    if np.abs(coords[-1] -
              _data_strip_bimedian_height) < 0.5 * _data_strip_thickness:
        velocity[0] = _data_strip_velocity
    else:
        velocity[0] = _data_background_velocity

    one_over_two_sigma_squared = 0.5 / (_data_perturb_width)**2
    strip_lower_bound = (_data_strip_bimedian_height -
                         0.5 * _data_strip_thickness)
    strip_upper_bound = (_data_strip_bimedian_height +
                         0.5 * _data_strip_thickness)
    velocity[-1] = (np.exp(-one_over_two_sigma_squared *
                           (coords[-1] - strip_lower_bound)**2) +
                    np.exp(-one_over_two_sigma_squared *
                           (coords[-1] - strip_upper_bound)**2))
    velocity[-1] *= _data_perturb_amplitude * np.sin(4.0 * np.pi * coords[0])
    return np.asarray(velocity)


def data_momentum_density(face_mesh_velocity, outward_directed_normal_covector,
                          coords, time, dim):
    return data_mass_density(
        face_mesh_velocity,
        outward_directed_normal_covector, coords, time, dim) * data_velocity(
            face_mesh_velocity, outward_directed_normal_covector, coords, time,
            dim)


def data_pressure(face_mesh_velocity, outward_directed_normal_covector, coords,
                  time, dim):
    return _data_pressure


def data_specific_internal_energy(face_mesh_velocity,
                                  outward_directed_normal_covector, coords,
                                  time, dim):
    return 1.0 / (_data_adiabatic_index - 1.0) * data_pressure(
        face_mesh_velocity, outward_directed_normal_covector,
        coords, time, dim) / data_mass_density(
            face_mesh_velocity, outward_directed_normal_covector, coords, time,
            dim)


def data_energy_density(face_mesh_velocity, outward_directed_normal_covector,
                        coords, time, dim):
    velocity = data_velocity(face_mesh_velocity,
                             outward_directed_normal_covector, coords, time,
                             dim)
    int_energy = data_specific_internal_energy(
        face_mesh_velocity, outward_directed_normal_covector, coords, time,
        dim)
    return data_mass_density(
        face_mesh_velocity, outward_directed_normal_covector, coords, time,
        dim) * (0.5 * np.dot(velocity, velocity) + int_energy)


def data_flux_mass_density(face_mesh_velocity,
                           outward_directed_normal_covector, coords, time,
                           dim):
    return data_momentum_density(face_mesh_velocity,
                                 outward_directed_normal_covector, coords,
                                 time, dim)


def data_flux_momentum_density(face_mesh_velocity,
                               outward_directed_normal_covector, coords, time,
                               dim):
    velocity = data_velocity(face_mesh_velocity,
                             outward_directed_normal_covector, coords, time,
                             dim)
    pressure = data_pressure(face_mesh_velocity,
                             outward_directed_normal_covector, coords, time,
                             dim)
    return flux.momentum_density_flux_impl(
        data_momentum_density(face_mesh_velocity,
                              outward_directed_normal_covector, coords, time,
                              dim),
        data_energy_density(face_mesh_velocity,
                            outward_directed_normal_covector, coords, time,
                            dim), velocity, pressure)


def data_flux_energy_density(face_mesh_velocity,
                             outward_directed_normal_covector, coords, time,
                             dim):
    velocity = data_velocity(face_mesh_velocity,
                             outward_directed_normal_covector, coords, time,
                             dim)
    pressure = data_pressure(face_mesh_velocity,
                             outward_directed_normal_covector, coords, time,
                             dim)
    return flux.energy_density_flux_impl(
        data_momentum_density(face_mesh_velocity,
                              outward_directed_normal_covector, coords, time,
                              dim),
        data_energy_density(face_mesh_velocity,
                            outward_directed_normal_covector, coords, time,
                            dim), velocity, pressure)
