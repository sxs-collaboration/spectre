# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import Fluxes, Sources


def tilde_e_flux(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                 current_density, kappa_psi, kappa_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 extrinsic_curvature, d_lapse, d_shift, d_spatial_metric):
    return Fluxes.tilde_e_flux(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                               current_density, lapse, shift,
                               sqrt_det_spatial_metric, spatial_metric,
                               inv_spatial_metric)


def tilde_b_flux(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                 current_density, kappa_psi, kappa_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 extrinsic_curvature, d_lapse, d_shift, d_spatial_metric):
    return Fluxes.tilde_b_flux(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                               current_density, lapse, shift,
                               sqrt_det_spatial_metric, spatial_metric,
                               inv_spatial_metric)


def tilde_psi_flux(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                   current_density, kappa_psi, kappa_phi, lapse, shift,
                   sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                   extrinsic_curvature, d_lapse, d_shift, d_spatial_metric):
    return Fluxes.tilde_psi_flux(tilde_e, tilde_b, tilde_psi, tilde_phi,
                                 tilde_q, current_density, lapse, shift,
                                 sqrt_det_spatial_metric, spatial_metric,
                                 inv_spatial_metric)


def tilde_phi_flux(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                   current_density, kappa_psi, kappa_phi, lapse, shift,
                   sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                   extrinsic_curvature, d_lapse, d_shift, d_spatial_metric):
    return Fluxes.tilde_phi_flux(tilde_e, tilde_b, tilde_psi, tilde_phi,
                                 tilde_q, current_density, lapse, shift,
                                 sqrt_det_spatial_metric, spatial_metric,
                                 inv_spatial_metric)


def tilde_q_flux(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                 current_density, kappa_psi, kappa_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 extrinsic_curvature, d_lapse, d_shift, d_spatial_metric):
    return Fluxes.tilde_q_flux(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                               current_density, lapse, shift,
                               sqrt_det_spatial_metric, spatial_metric,
                               inv_spatial_metric)


def source_tilde_e(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                   current_density, kappa_psi, kappa_phi, lapse, shift,
                   sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                   extrinsic_curvature, d_lapse, d_shift, d_spatial_metric):
    return Sources.source_tilde_e(tilde_e, tilde_b, tilde_psi, tilde_phi,
                                  tilde_q, current_density, kappa_psi,
                                  kappa_phi, lapse, d_lapse, d_shift,
                                  d_spatial_metric, inv_spatial_metric,
                                  sqrt_det_spatial_metric, extrinsic_curvature)


def source_tilde_b(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                   current_density, kappa_psi, kappa_phi, lapse, shift,
                   sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                   extrinsic_curvature, d_lapse, d_shift, d_spatial_metric):
    return Sources.source_tilde_b(tilde_e, tilde_b, tilde_psi, tilde_phi,
                                  tilde_q, current_density, kappa_psi,
                                  kappa_phi, lapse, d_lapse, d_shift,
                                  d_spatial_metric, inv_spatial_metric,
                                  sqrt_det_spatial_metric, extrinsic_curvature)


def source_tilde_psi(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                     current_density, kappa_psi, kappa_phi, lapse, shift,
                     sqrt_det_spatial_metric, spatial_metric,
                     inv_spatial_metric, extrinsic_curvature, d_lapse, d_shift,
                     d_spatial_metric):
    return Sources.source_tilde_psi(tilde_e, tilde_b, tilde_psi, tilde_phi,
                                    tilde_q, current_density, kappa_psi,
                                    kappa_phi, lapse, d_lapse, d_shift,
                                    d_spatial_metric, inv_spatial_metric,
                                    sqrt_det_spatial_metric,
                                    extrinsic_curvature)


def source_tilde_phi(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                     current_density, kappa_psi, kappa_phi, lapse, shift,
                     sqrt_det_spatial_metric, spatial_metric,
                     inv_spatial_metric, extrinsic_curvature, d_lapse, d_shift,
                     d_spatial_metric):
    return Sources.source_tilde_phi(tilde_e, tilde_b, tilde_psi, tilde_phi,
                                    tilde_q, current_density, kappa_psi,
                                    kappa_phi, lapse, d_lapse, d_shift,
                                    d_spatial_metric, inv_spatial_metric,
                                    sqrt_det_spatial_metric,
                                    extrinsic_curvature)


def source_tilde_q(tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q,
                   current_density, kappa_psi, kappa_phi, lapse, shift,
                   sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                   extrinsic_curvature, d_lapse, d_shift, d_spatial_metric):
    return 0.0 * lapse
