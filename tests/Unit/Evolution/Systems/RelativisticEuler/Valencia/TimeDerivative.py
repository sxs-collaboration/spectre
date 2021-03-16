# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import Sources, Fluxes


def source_tilde_d(tilde_d, tilde_tau, tilde_s, lapse, shift,
                   sqrt_det_spatial_metric, pressure, spatial_velocity,
                   d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
                   extrinsic_curvature, spatial_metric):
    return 0.0 * tilde_d


def source_tilde_tau(tilde_d, tilde_tau, tilde_s, lapse, shift,
                     sqrt_det_spatial_metric, pressure, spatial_velocity,
                     d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
                     extrinsic_curvature, spatial_metric):
    return Sources.source_tilde_tau(tilde_d, tilde_tau, tilde_s,
                                    spatial_velocity, pressure, lapse, d_lapse,
                                    d_shift, d_spatial_metric,
                                    inv_spatial_metric,
                                    sqrt_det_spatial_metric,
                                    extrinsic_curvature)


def source_tilde_s(tilde_d, tilde_tau, tilde_s, lapse, shift,
                   sqrt_det_spatial_metric, pressure, spatial_velocity,
                   d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
                   extrinsic_curvature, spatial_metric):
    return Sources.source_tilde_s(tilde_d, tilde_tau, tilde_s,
                                  spatial_velocity, pressure, lapse, d_lapse,
                                  d_shift, d_spatial_metric,
                                  inv_spatial_metric, sqrt_det_spatial_metric,
                                  extrinsic_curvature)


def tilde_d_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                 sqrt_det_spatial_metric, pressure, spatial_velocity, d_lapse,
                 d_shift, d_spatial_metric, inv_spatial_metric,
                 extrinsic_curvature, spatial_metric):
    return Fluxes.tilde_d_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                               sqrt_det_spatial_metric, pressure,
                               spatial_velocity)


def tilde_tau_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                   sqrt_det_spatial_metric, pressure, spatial_velocity,
                   d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
                   extrinsic_curvature, spatial_metric):
    return Fluxes.tilde_tau_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                                 sqrt_det_spatial_metric, pressure,
                                 spatial_velocity)


def tilde_s_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                 sqrt_det_spatial_metric, pressure, spatial_velocity, d_lapse,
                 d_shift, d_spatial_metric, inv_spatial_metric,
                 extrinsic_curvature, spatial_metric):
    return Fluxes.tilde_s_flux(tilde_d, tilde_tau, tilde_s, lapse, shift,
                               sqrt_det_spatial_metric, pressure,
                               spatial_velocity)
