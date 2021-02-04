# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import PointwiseFunctions.AnalyticSolutions.GeneralRelativity.GaugeWave as gw
import PointwiseFunctions.GeneralRelativity.ComputeSpacetimeQuantities as gr
import PointwiseFunctions.GeneralRelativity.ComputeGhQuantities as gh


def error(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, coords, interior_gamma1,
          interior_gamma2, time, dim):
    return None


_amplitude = 0.2
_wavelength = 10.0


def lapse(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, coords, interior_gamma1,
          interior_gamma2, time, dim):
    return gw.gauge_wave_lapse(coords, time, _amplitude, _wavelength)


def shift(face_mesh_velocity, outward_directed_normal_covector,
          outward_directed_normal_vector, coords, interior_gamma1,
          interior_gamma2, time, dim):
    return gw.gauge_wave_shift(coords, time, _amplitude, _wavelength)


def spacetime_metric(face_mesh_velocity, outward_directed_normal_covector,
                     outward_directed_normal_vector, coords, interior_gamma1,
                     interior_gamma2, time, dim):
    return gr.spacetime_metric(
        gw.gauge_wave_lapse(coords, time, _amplitude, _wavelength),
        gw.gauge_wave_shift(coords, time, _amplitude, _wavelength),
        gw.gauge_wave_spatial_metric(coords, time, _amplitude, _wavelength))


def phi(face_mesh_velocity, outward_directed_normal_covector,
        outward_directed_normal_vector, coords, interior_gamma1,
        interior_gamma2, time, dim):
    lapse = gw.gauge_wave_lapse(coords, time, _amplitude, _wavelength)
    shift = gw.gauge_wave_shift(coords, time, _amplitude, _wavelength)
    spatial_metric = gw.gauge_wave_spatial_metric(coords, time, _amplitude,
                                                  _wavelength)
    deriv_lapse = gw.gauge_wave_d_lapse(coords, time, _amplitude, _wavelength)
    deriv_shift = gw.gauge_wave_d_shift(coords, time, _amplitude, _wavelength)
    deriv_spatial_metric = gw.gauge_wave_d_spatial_metric(
        coords, time, _amplitude, _wavelength)
    return gh.phi(lapse, deriv_lapse, shift, deriv_shift, spatial_metric,
                  deriv_spatial_metric)


def pi(face_mesh_velocity, outward_directed_normal_covector,
       outward_directed_normal_vector, coords, interior_gamma1,
       interior_gamma2, time, dim):
    lapse = gw.gauge_wave_lapse(coords, time, _amplitude, _wavelength)
    shift = gw.gauge_wave_shift(coords, time, _amplitude, _wavelength)
    spatial_metric = gw.gauge_wave_spatial_metric(coords, time, _amplitude,
                                                  _wavelength)
    dt_lapse = gw.gauge_wave_dt_lapse(coords, time, _amplitude, _wavelength)
    dt_shift = gw.gauge_wave_dt_shift(coords, time, _amplitude, _wavelength)
    dt_spatial_metric = gw.gauge_wave_dt_spatial_metric(
        coords, time, _amplitude, _wavelength)
    return gh.pi(
        lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric,
        phi(face_mesh_velocity, outward_directed_normal_covector,
            outward_directed_normal_vector, coords, interior_gamma1,
            interior_gamma2, time, dim))


def constraint_gamma1(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, interior_gamma1,
                      interior_gamma2, time, dim):
    assert interior_gamma1 >= 0.0
    return interior_gamma1


def constraint_gamma2(face_mesh_velocity, outward_directed_normal_covector,
                      outward_directed_normal_vector, coords, interior_gamma1,
                      interior_gamma2, time, dim):
    assert interior_gamma2 >= 0.0
    return interior_gamma2
